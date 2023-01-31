# Author: Daniel Canedo
#
# Summary: Breaks the TIF image into smaller images, uses the YOLOv5 models for inference, performs late fusion, stores
#		   images that had detected objects, transforms the detected objects pixel coordinates to real-world coordinates.
#
# [!] Still early in development
# TODO: 50% step nms gis bbs, dont use LOF on original sites

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
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
import laspy
import pickle
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import pyqtree

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None


def nms(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	boxes = np.array(boxes)
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,2]
	x2 = boxes[:,1]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


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

	return (xMin, xMax, yMin, yMax)


def pointCloud(spindex, validationModel, pointClouds, cropExtent, className, bb):
	tmp = ''
	for cloud in os.listdir(pointClouds):
		tmp = pointClouds + '/' + cloud
		break


	# Creates empty .las file to later populate it with points
	with laspy.open(tmp) as f:
		w = laspy.open('tmp.las', mode='w', header = f.header)
		w.close()

	count = 0
	# Checks if there is an overlap with the cropped image and the point cloud
	matches = spindex.intersect((bb[0], bb[2], bb[1], bb[3]))

	# Iterates over the matched point clouds
	with laspy.open('tmp.las', mode = 'a') as w:
		for match in matches:
			with laspy.open(match) as f:
				# Appends the points of the overlapping region to the previously created .las file
				las = f.read()          
				x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
				mask = (x >= bb[0]) & (x <= bb[1]) & (y >= bb[2]) & (y <= bb[3])
				if True in mask:
					roi = las.points[las.classification == 2][mask]
					w.append_points(roi)
					count += 1
		
	if count > 0:
		xyz = las_utils.read_las_xyz('tmp.las')
		#FEATURE_NAMES = ['linearity', 'planarity', 'surface_variation', 'sphericity']
		features = compute_features(xyz, search_radius=3)#, feature_names = ['linearity', 'planarity', 'surface_variation', 'sphericity'])
		
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
				median = np.median(stats[i])
				var = np.var(stats[i])
				stdev = np.std(stats[i])
				cov = np.cov(stats[i])
				X += [median, stdev, var, cov]
				#X.append(cov)
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

	imgsz=(resolution, resolution)  # inference size (pixels)
	conf_thres=0.25  # confidence threshold
	iou_thres=0.45  # NMS IOU threshold
	max_det=1000  # maximum detections per image
	agnostic_nms=False
	classes=None
	bs = 1
	device = "0"
	device = select_device(device)

	weights = 'paper.pt'

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
	# Load YOLO model
	model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
	stride, names, pt = model.stride, model.names, model.pt
	imgsz = check_img_size(imgsz, s=stride)  # check image size
	model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
	seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

	print(names)
	aux = {}
	for name in names.values():
		aux[name] = []

	print(aux)

	pointClouds = 'LAS'
	spindex = pyqtree.Index(bbox=(0, 0, 100, 100))
	for cloud in os.listdir(pointClouds):
		with laspy.open(pointClouds + '/' + cloud) as f:
			spindex.insert(pointClouds + '/' + cloud, (f.header.x_min, f.header.y_min, f.header.x_max, f.header.y_max))
	
	# Iterates over the images
	detections = 0
	validated = 0
	TP = 0
	FP = 0
	FN = 0
	TPval = 0
	FPval = 0
	FNval = 0
	for image in images:

		# Load image
		if image.endswith('Viana.tif'):
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
			annotatedBbs = []
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
					annotatedBbs.append(bb)

			yoloDetections = []
			validatedDetections = []
			for i in range(0, (width-resolution), resolution//2):
				for j in range(0, (height - resolution), resolution//2):
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
						boxes = 0
						for bb in bbs:
							if intersection(bb, cropExtent):
								boxes += 1
								xMin = int(map(bb[0], i, i+resolution, 0, resolution))
								xMax = int(map(bb[1], i, i+resolution, 0, resolution))
								yMin = int(map(bb[2], j, j+resolution, 0, resolution))
								yMax = int(map(bb[3], j, j+resolution, 0, resolution))
								drawnBbs.append((xMin,xMax, yMin, yMax))
								cv2.rectangle(displayImg, (xMin, yMin), (xMax,yMax), (255,0,0), 2)
						if boxes > 0:
							with dt[0]:
								im = letterbox(croppedImg, imgsz, stride=stride, auto=True)[0]
								# Convert
								im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
								im = np.ascontiguousarray(im)
								im = torch.from_numpy(im).to(model.device)
								im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
								im /= 255  # 0 - 255 to 0.0 - 1.0
								if len(im.shape) == 3:
									im = im[None]  # expand for batch dim

							with dt[1]:
								pred = model(im, augment=False, visualize=False)

							# NMS
							with dt[2]:
								pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

							
							for x, det in enumerate(pred):
								if len(det):
									# Rescale boxes from img_size to im0 size
									det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], displayImg.shape).round()




									for *xyxy, conf, cls in reversed(det):
										cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 1)
										GISbb = convert2GIS(xyxy, cropExtent, width, height, resolution, xMinImg, yMinImg, xMaxImg, yMaxImg)
										lbr = LBR(roiPolygons, GISbb)
										print(lbr)
										yoloDetections.append(GISbb)
										
										
										

										if lbr:
											annotated = False
											for b in drawnBbs:
												if getIou(b, (int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3]))) > iou_thres:
													annotated = True
													break

											if annotated == True:
												color = (0,255,0)
												print('Detection validated')
												cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)												
												validatedDetections.append(GISbb)

											else:

												c = int(cls)  # integer class
												className = names[c]
												validation = pointCloud(spindex, validationModel, pointClouds, cropExtent, className, GISbb)
											
												if validation == True:
													#boxes += 1
													color = (0,255,0)
													print('Detection validated')
													cv2.rectangle(displayImg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)
													
													validatedDetections.append(GISbb)												


									
									

						
						#if boxes > 0:
							#cv2.imshow("Cropped Image", displayImg)
							#cv2.waitKey(0)

			img.close()

			finalYolo = nms(yoloDetections, iou_thres)
			finalValidated = nms(validatedDetections, iou_thres)
			

			detections += len(finalYolo)
			validated += len(finalValidated)

			
			for d in finalYolo:
				annotated = False
				for b in annotatedBbs:  
					if getIou(d, b) > 0.2:
						annotated = True
						break
				if annotated == True:
					TP += 1
				else: 
					FP += 1
			FN = len(annotatedBbs) - TP

			for d in finalValidated:
				xMin, xMax, yMin, yMax = d
				strGISbb = '((' + str(xMin) + ' ' + str(yMin) + ', ' + str(xMin) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMax) + ', ' + str(xMax) + ' ' + str(yMin) + '))'
				aux[className].append(strGISbb)
				annotated = False
				for b in annotatedBbs:  
					if getIou(d, b) > 0.2:
						annotated = True
						break
				if annotated == True:
					TPval += 1
				else: 
					FPval += 1
			FNval = len(annotatedBbs) - TPval


				



	print('[===========================]')
	print('Detections:', detections)
	print('Validated detections:', validated)
	print('TP:', TP)
	print('FP:', FP)
	print('FN:', FN)
	print('TPval:', TPval)
	print('FPval:', FPval)
	print('FNval:', FNval)
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