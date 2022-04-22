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
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
Image.MAX_IMAGE_PIXELS = None

# Converts coordinates from GIS reference to image pixels
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	xB = min(boxA[1], boxB[1])
	yB = min(boxA[2], boxB[2])
	yA = max(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def main():
	resolution = 0
	while resolution <= 0:
		resolution = int(input("Image resolution: "))

	imgsz=resolution  # inference size (pixels)
	conf_thres=0.25  # confidence threshold
	iou_thres=0.45  # NMS IOU threshold
	max_det=1000  # maximum detections per image
	classes=None  # filter by class: --class 0, or --class 0 2 3
	agnostic_nms=False  # class-agnostic NMS
	cudnn.benchmark = True  # set True to speed up constant image size inference
	device = 'cpu'
	device = select_device(device)

	folder = filedialog.askdirectory(title = "Select the folder with the weights")
	weights = []
	for file in os.listdir(folder):
		if file.endswith(".pt"):
			weights.append(folder + '/' + file)


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

	# Saves all the images path to a list
	database = {}
	
	folders = []
	for f in os.listdir(os.path.split(folder)[0]):
		folders.append(os.path.split(folder)[0] + '/' + f)


	print(folders)

	for f in folders:
		weight = [path for path in weights if os.path.split(f)[1] in path][0]
		model = attempt_load(weight, map_location=device)
		database[f] = {}
		database[f]['Model'] = model

	# Checks if the list is empty = no image was found in the selected folder
	if not database:
		sys.exit("Selected folder is empty.")

	if not os.path.exists('results/'):
		os.makedirs('results/')

	keys = list(database.keys())


	endEarly = 0
	# Iterates over the images
	for image in os.listdir(keys[0]):
		if endEarly > 0:
			break

		# Load image
		if image.endswith('.tif'):
			print('Processing ' + keys[0] + '/' +image)
			img = Image.open(keys[0] + '/' + image).convert('RGB') 
			geoRef = rasterio.open(keys[0] + '/' + image)
			width, height = img.size
			
			stride = int(database[keys[0]]['Model'].stride.max())  # model stride
			names = database[keys[0]]['Model'].module.names if hasattr(database[keys[0]], 'module') else database[keys[0]]['Model'].names  # get class names

			detections = {}

			# Breaks the image into blocks with a sliding window of resolution/2
			#for i in range(0, width - resolution, resolution//2):
			#	for j in range(0, height - resolution, resolution//2):
			for i in range(0, 640, resolution//2):
				for j in range(0, (height - resolution), resolution//2):
					coords = '('+ str(0+i) + ',' + str(resolution+i) + ',' + str(0+j) + ',' + str(resolution+j) + ')'
					
					detections[coords] = {}
					detections[coords]['bb'] = []
					detections[coords]['label'] = []
					detections[coords]['conf'] = []
					detections[coords]['data'] = []

					croppedOriginalImg = img.crop((0+i, 0+j, resolution+i, resolution+j))
					croppedImg = np.array(croppedOriginalImg)
					displayImg = croppedImg.copy()
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

					pred = database[keys[0]]['Model'](croppedImg)[0]
					pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

					for x, det in enumerate(pred):
						annotator = Annotator(displayImg, line_width=2, example=str(names))
						if len(det):
							# Rescale boxes from img_size to im0 size
							det[:, :4] = scale_coords(croppedImg.shape[2:], det[:, :4], displayImg.shape).round()


							for *xyxy, conf, cls in reversed(det):
								c = int(cls)  # integer class
								label = f'{names[c]} {conf:.2f}'
								annotator.box_label(xyxy, label, color=colors(c, True))
								displayImg = annotator.result()

								detections[coords]['bb'].append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
								detections[coords]['label'].append(c)
								detections[coords]['conf'].append(float(conf))
								detections[coords]['data'].append(os.path.split(keys[0])[1])
								detections[coords]['crop'] = (0+i, resolution+i, 0+j, resolution+j)


					#cv2.imshow("Cropped Image", displayImg)
					#cv2.waitKey(30)

			img.close()


			

			for key in keys[1:]:
				newImage = key + '/' + image

				if os.path.exists(newImage):
					
					print('Processing ' + newImage)
					img = Image.open(newImage).convert('RGB') 
					
					stride = int(database[key]['Model'].stride.max())  # model stride
					names = database[key]['Model'].module.names if hasattr(database[key], 'module') else database[key]['Model'].names  # get class names

					# Breaks the image into blocks with a sliding window of resolution/2
					#for i in range(0, width - resolution, resolution//2):
					#	for j in range(0, height - resolution, resolution//2):
					for i in range(0, 640, resolution//2):
						for j in range(0, (height - resolution), resolution//2):
							coords = '('+ str(0+i) + ',' + str(resolution+i) + ',' + str(0+j) + ',' + str(resolution+j) + ')'
							
							if coords in detections.keys() == False:
								detections[coords] = {}
								detections[coords]['bb'] = []
								detections[coords]['label'] = []
								detections[coords]['conf'] = []
								detections[coords]['data'] = []

							croppedOriginalImg = img.crop((0+i, 0+j, resolution+i, resolution+j))
							croppedImg = np.array(croppedOriginalImg)
							displayImg = croppedImg.copy()
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

							pred = database[key]['Model'](croppedImg)[0]
							pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

							for x, det in enumerate(pred):
								annotator = Annotator(displayImg, line_width=2, example=str(names))
								if len(det):
									# Rescale boxes from img_size to im0 size
									det[:, :4] = scale_coords(croppedImg.shape[2:], det[:, :4], displayImg.shape).round()


									for *xyxy, conf, cls in reversed(det):
										c = int(cls)  # integer class
										label = f'{names[c]} {conf:.2f}'
										annotator.box_label(xyxy, label, color=colors(c, True))
										displayImg = annotator.result()
										if coords in detections.keys() == False:
											detections[coords]['bb'].append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
											detections[coords]['label'].append(c)
											detections[coords]['conf'].append(float(conf))
											detections[coords]['data'].append(os.path.split(key)[1])
											detections[coords]['crop'] = (0+i, resolution+i, 0+j, resolution+j)
										else:
											for x in range(len(detections[coords]['bb'])):
												if iou(detections[coords]['bb'][x], (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))) >= 0.3:
													detections[coords]['conf'] += float(conf)
												else:
													detections[coords]['bb'].append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
													detections[coords]['label'].append(c)
													detections[coords]['conf'].append(float(conf))
													detections[coords]['data'].append(os.path.split(key)[1])

							#cv2.imshow("Cropped Image", displayImg)
							#cv2.waitKey(30)

					img.close()



			img = Image.open(folder + '/' + image)

			coords = list(detections.keys())


			for coord in coords:
				bbox = []
				labels = []
				data = []
				for i in range(len(detections[coord]['conf'])):
					if detections[coord]['conf'][i]/len(keys) >= 0.001:
						bbox.append(detections[coord]['bb'][i])
						labels.append(detections[coord]['label'][i])
						data.append(detections[coord]['data'][i])


				if bbox:
					crop = detections[coord]['crop']
					croppedImg = img.crop((crop[0], crop[2], crop[1], crop[3]))
					img1 = ImageDraw.Draw(croppedImg)
					for i in range (len(bbox)):
						if data[i] == 'LRM':
							img1.rectangle([bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]], outline='Red', width = 2)
						else:
							img1.rectangle([bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]], outline='Blue', width = 2)
						
					imgName = image.split('.')[0] + '-' + os.path.split(folder)[1] + '('+ str(crop[0]) + ',' + str(crop[1]) + ',' + str(crop[2]) + ',' + str(crop[3]) + ')'
					croppedImg.save('results/' + imgName + '.png')

			img.close()

		endEarly += 1


if __name__ == "__main__":
	main()