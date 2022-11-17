# Author: Daniel Canedo
#
# Summary: Breaks the TIF image into smaller images, uses the YOLOv5 models for inference, performs late fusion, stores
#		   images that had detected objects, transforms the detected objects pixel coordinates to real-world coordinates.
#
# [!] Still early in development

import os
import sys
import math
import csv
import numpy as np 
import cv2
from tkinter import filedialog
import rasterio
from PIL import Image, ImageDraw
import torch
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
import laspy
import pickle
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from shapely.geometry import Point, Polygon
from shapely.wkt import loads

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

def testpoly2bb(row, xMinImg, xMaxImg, yMaxImg, yMinImg, width, height):
	raw = row.split(", ")
	raw[0] = raw[0].strip(raw[0][:raw[0].find('(')] + '(((')
	raw[len(raw)-1] = raw[len(raw)-1][:len(raw[len(raw)-1])-2]

	bbs = []
	xPoints = []
	yPoints = []
	for i in range(0, len(raw)):
		if raw[i][len(raw[i])-1] != ')':
			if raw[i][0] != '(':
				xPoints.append(float(raw[i].split(" ")[0]))
			else:
				xPoints.append(float(raw[i].split(" ")[0].strip('((')))
			yPoints.append(float(raw[i].split(" ")[1]))
		else:
			xPoints.append(float(raw[i].split(" ")[0]))
			yPoints.append(float(raw[i].split(" ")[1].strip(')')))

			xMin = min(xPoints)
			xMax = max(xPoints)
			yMin = min(yPoints)
			yMax = max(yPoints)

			# The bounding box must be within the image limits
			if xMin < xMinImg:
				xMin = xMinImg
			if xMax > xMaxImg:
				xMax = xMaxImg
			if yMin < yMinImg:
				yMin = yMinImg
			if yMax > yMaxImg:
				yMax = yMaxImg

			# Maps coordinates from GIS reference to image pixels
			xMinBb = int(map(xMin, xMinImg, xMaxImg, 0, width))
			xMaxBb = int(map(xMax, xMinImg, xMaxImg, 0, width))
			yMaxBb = int(map(yMin, yMinImg, yMaxImg, height, 0))
			yMinBb = int(map(yMax, yMinImg, yMaxImg, height, 0))
			xPoints = []
			yPoints = []

			bbs.append((xMinBb, xMaxBb, yMinBb, yMaxBb))
	return bbs

def poly2bb(row, xMinImg, xMaxImg, yMaxImg, yMinImg):
	raw = row.split(",")
	raw[0] = raw[0].strip(raw[0][:raw[0].find('(')] + '(((')
	raw[len(raw)-1] = raw[len(raw)-1].strip(')))')
	xPoints = []
	yPoints = []
	for i in range(0, len(raw)):
		xPoints.append(float(raw[i].split(" ")[0]))
		yPoints.append(float(raw[i].split(" ")[1]))

	xMin = min(xPoints)
	xMax = max(xPoints)
	yMin = min(yPoints)
	yMax = max(yPoints)

	# The bounding box must be within the image limits
	if xMin < xMinImg:
		xMin = xMinImg
	if xMax > xMaxImg:
		xMax = xMaxImg
	if yMin < yMinImg:
		yMin = yMinImg
	if yMax > yMaxImg:
		yMax = yMaxImg

	return (xMin, xMax, yMin, yMax)

# Converts coordinates from GIS reference to image pixels
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

def getIou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[2], boxB[2])
	xB = min(boxA[1], boxB[1])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg):

	xMin = map(int(xyxy[0]), 0, resolution, cropExtent[0], cropExtent[1])
	xMax = map(int(xyxy[2]), 0, resolution, cropExtent[0], cropExtent[1])
	yMin = map(int(xyxy[3]), 0, resolution, cropExtent[2], cropExtent[3])
	yMax = map(int(xyxy[1]), 0, resolution, cropExtent[2], cropExtent[3])

	xMin = map(xMin, 0, width, xMinImg, xMaxImg)
	xMax = map(xMax, 0, width, xMinImg, xMaxImg)
	yMax = map(yMax, height, 0, yMinImg, yMaxImg)
	yMin = map(yMin, height, 0, yMinImg, yMaxImg)

	return (xMin, xMax, yMin, yMax), '((' + str(xMin) + ' ' + str(yMin) + ', ' + str(xMin) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMin) + '))'


def pointCloud(validationModel, pointClouds, cropExtent, className, bb):
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break

	# Creates empty .las file to later populate it with points
	with laspy.open(tmp) as f:
		w = laspy.open('tmp.las', mode='w', header = f.header)
		w.close()

	count = 0
	# Iterates over the point clouds
	with laspy.open('tmp.las', mode = 'a') as w:
		for cloud in os.listdir(pointClouds):
			with laspy.open(pointClouds + '/' + cloud) as f:

				# Checks if there is an overlap with the cropped image and the point cloud
				if bb[0] <= f.header.x_max and bb[1] >= f.header.x_min and bb[2] <= f.header.y_max and bb[3] >= f.header.y_min:
					# Appends the points of the overlapping region to the previously created .las file
					las = f.read()          
					x, y = las.points.x.copy(), las.points.y.copy()
					mask = (x >= bb[0]) & (x <= bb[1]) & (y >= bb[2]) & (y <= bb[3])
					roi = las.points[mask]
					w.append_points(roi)
					count += 1
		
	if count > 0:
		xyz = las_utils.read_las_xyz('tmp.las')
		#FEATURE_NAMES = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']
		features = compute_features(xyz, search_radius=3)#, feature_names = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality'])
		
		if np.isnan(features).any() == False:

			stats = {}
			for i in FEATURE_NAMES:
				stats[i] = []
			
			for feature in features:
				for i in range(len(FEATURE_NAMES)):
					stats[FEATURE_NAMES[i]].append(feature[i])

			X = []
			for i in FEATURE_NAMES:		
				mean = np.mean(stats[i])
				stdev = np.std(stats[i])
				X += [mean,stdev]
				#print(i + ': ' + str(mean) + ' - ' + str(stdev))



			#X += list(np.max(xyz, axis=0)-np.min(xyz, axis=0))
			#X += [np.mean(xyz, axis=0)[2], np.std(xyz, axis=0)[2]]
			
			
			os.remove('tmp.las')
			if validationModel.predict([X]) == -1:
				return False
			else:
				return True

	
	return -1



def intersection(bb, crop):
    return not (bb[1] < crop[0] or bb[0] > crop[1] or bb[3] < crop[2] or bb[2] > crop[3])


# TODO: update poly2bb
def parsePoly(row):
	raw = row.split(",")
	raw[0] = raw[0].strip(raw[0][:raw[0].find('(')] + '(((')
	raw[len(raw)-1] = raw[len(raw)-1][:len(raw[len(raw)-1])-2]

	polygons = []
	tmp = []
	for i in range(0, len(raw)):
		if raw[i][len(raw[i])-1] != ')':
			if raw[i][0] != '(':
				xPoint = float(raw[i].split(" ")[0])
			else:
				xPoint = float(raw[i].split(" ")[0].strip('('))
			yPoint = float(raw[i].split(" ")[1])
			tmp.append((xPoint,yPoint))
		else:
			xPoint = float(raw[i].split(" ")[0])
			yPoint = float(raw[i].split(" ")[1].strip(')'))
			tmp.append((xPoint,yPoint))
			polygons.extend(tmp)
			tmp = []
	return polygons


def LBRroi(polygons, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	intersection = []
	for polygon in polygons:
		if polygon.intersects(p):
			intersection.append(polygon.intersection(p))
	return intersection

def LBR(roi, bb):
	p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
	for polygon in roi:
		if polygon.intersects(p):
			return True
	return False

def main():
	resolution = 640

	imgsz=resolution  # inference size (pixels)
	conf_thres=0.25  # confidence threshold
	iou_thres=0.45  # NMS IOU threshold
	max_det=1000  # maximum detections per image
	classes=None  # filter by class: --class 0, or --class 0 2 3
	agnostic_nms=False  # class-agnostic NMS
	cudnn.benchmark = True  # set True to speed up constant image size inference
	device = 'cpu'
	device = select_device(device)

	weights = 'best.pt'

	polygonsCsv = 'Segmentation.csv'

	polygons = []
	with open(polygonsCsv) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			p = loads(row["WKT"])
			polygons.append(p)


	# Gets the path to the directory where the test images are located
	# For example:
	#
	# dataset/					
	# |-- DTMs/
	# 	  |-- image1.tif
	# 	  |-- image2.tif
	# 	  |-- ...
	#
	# |-- LRMs/              <- This folder must be selected!
	# 	  |-- images
	# 	  |-- image1.tif
	# 	  |-- image2.tif
	# 	  |-- ...

	folder = filedialog.askdirectory(title = "Select a folder with test images")


	print(folder)
	images = []
	for f in os.listdir(folder):
		images.append(folder + '/' + f)


	print(images)
	validationModel = pickle.load(open('pointCloud.sav', 'rb'))
	model = attempt_load(weights, map_location=device)
	stride = int(model.stride.max())  # model stride
	names = model.module.names if hasattr(model, 'module') else model.names  # get class names

	aux = {}
	for name in names:
		aux[name] = []

	pointClouds = 'LAS'
	# Iterates over the images
	validated = 0
	detections = 0
	for image in images:

		# Load image
		if image.endswith('.tif'):
			annotation = image.split('.')[0] + '.csv'
			print('Processing '  + image)
			img = Image.open(image).convert('RGB') 
			geoRef = rasterio.open(image)
			width, height = img.size

			# Parse corners of the image (GIS reference)
			#Left, Bottom, Right, Top
			xMinImg = geoRef.bounds[0]
			xMaxImg = geoRef.bounds[2]
			yMinImg = geoRef.bounds[1]
			yMaxImg = geoRef.bounds[3]

			bbs = []
			# Open the CSV file containing the object annotations
			with open(annotation) as csvfile:
				reader = csv.DictReader(csvfile)
				# This considers that polygons are under the column name "WKT" and labels are under the column name "Id"
				for row in reader:

					# Converts polygon to bounding box
					bb = poly2bb(row['WKT'], xMinImg, xMaxImg, yMaxImg, yMinImg)
					# Maps coordinates from GIS reference to image pixels
					xMin = int(map(bb[0], xMinImg, xMaxImg, 0, width))
					xMax = int(map(bb[1], xMinImg, xMaxImg, 0, width))
					yMax = int(map(bb[2], yMinImg, yMaxImg, height, 0))
					yMin = int(map(bb[3], yMinImg, yMaxImg, height, 0))

					bbs.append((xMin, xMax, yMin, yMax))


			#for i in range(0, (width-resolution), resolution):
			#	for j in range(0, (height - resolution), resolution):
			for i in range(13000, 21000, resolution):
				for j in range(9000, 16000, resolution):
					coords = '('+ str(0+i) + ',' + str(resolution+i) + ',' + str(0+j) + ',' + str(resolution+j) + ')'
					
					croppedOriginalImg = img.crop((i, j, resolution+i, resolution+j))
					cropExtent = [i, resolution+i, j, resolution+j]
					croppedImg = np.array(croppedOriginalImg)
					displayImg = croppedImg.copy()

					xMin = map(cropExtent[0], 0, width, xMinImg, xMaxImg)
					xMax = map(cropExtent[1], 0, width, xMinImg, xMaxImg)
					yMax = map(cropExtent[2], height, 0, yMinImg, yMaxImg)
					yMin = map(cropExtent[3], height, 0, yMinImg, yMaxImg)

					# get intersection polygon that will be used to check if detections are inside
					roiPolygons = LBRroi(polygons, (xMin,xMax,yMin,yMax))

					if len(roiPolygons) > 0:
						drawnBbs = []
						for bb in bbs:
							if intersection(bb, cropExtent):
								xMin = int(map(bb[0], i, i+resolution, 0, resolution))
								xMax = int(map(bb[1], i, i+resolution, 0, resolution))
								yMin = int(map(bb[2], j, j+resolution, 0, resolution))
								yMax = int(map(bb[3], j, j+resolution, 0, resolution))
								drawnBbs.append((xMin,xMax, yMin, yMax))
								cv2.rectangle(displayImg, (xMin, yMin), (xMax,yMax), (255,0,0), 2)

						croppedImg = letterbox(croppedImg, imgsz, stride=stride, auto=True)[0]
						# Convert
						croppedImg = croppedImg.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
						croppedImg = np.ascontiguousarray(croppedImg)
						# Inference
						# Save to a map, key = img coords, value = detections, confidence
						croppedImg = torch.from_numpy(croppedImg).to(device)
						croppedImg = croppedImg.float()
						croppedImg /= 255  # 0 - 255 to 0.0 - 1.0
						if len(croppedImg.shape) == 3:
							croppedImg = croppedImg[None]  # expand for batch dim

						pred = model(croppedImg)[0]
						pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

						boxes = 0
						for x, det in enumerate(pred):
							if len(det):
								# Rescale boxes from img_size to im0 size
								det[:, :4] = scale_coords(croppedImg.shape[2:], det[:, :4], displayImg.shape).round()


								for *xyxy, conf, cls in reversed(det):
									cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 1)
									GISbb, strGISbb = convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)
									lbr = LBR(roiPolygons, GISbb)
									print(lbr)
									if lbr:
										c = int(cls)  # integer class
										className = names[c]
										validation = pointCloud(validationModel, pointClouds, cropExtent, className, GISbb)
										if validation == False:
											annotated = False
											for b in drawnBbs:
												if getIou(b, (int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3]))) > 0.5:
													annotated = True
													break
											if annotated == False:
												detections += 1
												boxes += 1
												color = (0,0,255)	
												cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
										elif validation == True:
											annotated = False
											for b in drawnBbs:
												if getIou(b, (int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3]))) > 0.5:
													annotated = True
													break
											if annotated == False:
												detections += 1
												boxes += 1
												color = (0,255,0)
												print('Detection validated')
												validated += 1
												cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)


								
									# debug
									else:
										cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 4)

									aux[className].append(strGISbb)
									

						
						#if boxes > 0:
						cv2.imshow("Cropped Image", displayImg)
						cv2.waitKey(0)

			img.close()

		break

	print('[===========================]')
	print('Detections:', detections)
	print('Validated detections:', validated)
	print('[===========================]')


	f = open('detections.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(['WKT', 'Id'])
	data = {}
	for key in aux.keys():
		if len(aux[key]) != 0:
			data[key] = "MULTIPOLYGON ("			
			for i in range(len(aux[key])):
			
				if i != len(aux[key])-1:
					data[key] += aux[key][i] + ','
				else:
					data[key] += aux[key][i] + ')'
	
		writer.writerow([data[key], key])
	
	f.close()
	if os.path.isfile('tmp.las'):
		os.remove('tmp.las')

if __name__ == "__main__":
	main()