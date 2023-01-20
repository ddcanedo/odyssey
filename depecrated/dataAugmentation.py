# Author: Daniel Canedo
#
# Summary: Performs Data Augmentation on a selected dataset
# 
# Detailed Description:
# 1. The albumentations library was changed to return the bounding boxes regardless of their visibility after the augmentation.
#	 This was done because the min_visibility threshold is not working properly since it considers the ratio between the
#	 original bounding box area and the augmented bounding box area. For instance, this is not protected for image transformations
#	 that scales the images. 
#    Another important thing is that we want to discard augmented images which have partially visible objects and that is not possible
#    with the original library since it automatically deletes bounding boxes depending on the min_visibility threshold. We do not
#	 want to delete any bounding box so that we can check if they represent a partially visible object.
#
# 2. Augments images in which their objects cannot be partially visible to avoid getting hardly visible objects

import albumentations as A
import cv2
import os
from tkinter import filedialog

# Checks if the bounding box is totally outside the image, totally inside the image, or partially inside the image
def checkBbox(bb1, shape):
		#  bottom1 < top2		top1 > bottom2		right1 < left2		left1 > right2
    if (bb1[3] < 0) or (bb1[1] > shape[0]) or (bb1[2] < 0) or (bb1[0] > shape[1]):
    	return 1
    elif (bb1[0] >= 0) and (bb1[1] >= 0) and (bb1[2] <= shape[1]) and (bb1[3] <= shape[0]): 
    	return 2
    else:
    	return 3

def main():
	# Declares an augmentation pipeline
	transform = A.ReplayCompose([
	    A.Flip(p=0.75),
	    A.OneOf([A.ColorJitter(), A.HueSaturationValue(), A.RandomGamma(), A.RandomBrightnessContrast()], p=1),
	    A.RGBShift(p=0.5),
	    A.Perspective(p=0.75),
	], bbox_params=A.BboxParams(format='yolo', label_fields=['classLabels']))

	# Asks the user to select the dataset folder
	# For example:
	#
	# dataset/					<- This folder must be selected!
	# |-- DTMs/
	# 	  |-- images
	#     	  |-- train
	#         |-- val
	#	  |-- labels
	#     	  |-- train
	#         |-- val
	#
	# |-- LRMs/              
	# 	  |-- images
	#     	  |-- train
	#         |-- val
	#	  |-- labels
	#     	  |-- train
	#         |-- val
	#
	# |-- ...
	# Asks the user to select the dataset folder
	datasetPath = filedialog.askdirectory(title = "Path to the dataset") + '/'


	augmentations = 0
	trainval = ''

	while augmentations == 0:
		augmentations = int(input("How many augmentations per image: "))
	while trainval != "train" and trainval != "val":
		trainval = input("train/val: ")
	
	# Saves all the folders in the same directory
	labelsPath = {}
	for f in os.listdir(datasetPath):
		labelsPath[f] = datasetPath + f + '/labels/' + trainval + '/'

	imagesPath = {}
	for f in os.listdir(datasetPath):
		imagesPath[f] = datasetPath + f + '/images/' + trainval + '/'

	print('Performing Data Augmentation...')

	keys = list(imagesPath.keys())
	# Iterates over the images
	filename = []
	for file in os.listdir(labelsPath[keys[0]]):
		if file.split('.')[-1] == 'txt':
			filename.append(file.split('.')[0])

	for file in filename:
		# Gets the labels path associated to the image

		label = labelsPath[keys[0]] + '/' + file + '.txt'
		img  = imagesPath[keys[0]] + '/' + file + '.png'
		

		# Checks if image and label exist
		if(os.path.exists(label) and os.path.exists(img)):
			# Reads an image with OpenCV
			image = cv2.imread(img)

			# Names of the augmentations
			name = file + '-DA-'

			bboxes = []
			classLabels = []

			f = open(label, "r")
			for line in f:
				class_label = line.split(' ')[0]
				x = float(line.split(' ')[1])
				y = float(line.split(' ')[2])
				w = float(line.split(' ')[3])
				h = float(line.split(' ')[4])
				bboxes.append([x,y,w,h])
				classLabels.append(class_label)
			
			f.close()

			count = 0

			# Starts the Data Augmentation based on the pipeline created above
			while count < augmentations:
				transformed = transform(image=image, bboxes=bboxes, classLabels = classLabels)
				transformed_image = transformed['image']
				display_image = transformed_image.copy()
				transformed_bboxes = transformed['bboxes']
				transformed_classLabels = transformed['classLabels']
				
				valid = 2
				valid_size = 0
				# Iterates over the transformed bounding boxes
				for i in range(len(transformed_bboxes)):
					tl = (round((transformed_bboxes[i][0]-transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]-transformed_bboxes[i][3]/2)*transformed_image.shape[0]))
					br = (round((transformed_bboxes[i][0]+transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]+transformed_bboxes[i][3]/2)*transformed_image.shape[0]))

					cv2.rectangle(display_image, tl, br, (0,0,255), 1)

					bbox = (tl[0], tl[1], br[0], br[1])

					# Checks if the bounding box is partially visible, if so discards the augmentation
					valid = checkBbox(bbox, image.shape)
					if valid == 2:
						valid_size += 1
					if valid == 3:
						break

				# If there is at least 1 object 100% visible, saves the image and its labels
				if valid < 3 and valid_size > 0:
					cv2.imwrite(imagesPath[keys[0]] + name + str(count) + '.png' , transformed_image)
					with open(labelsPath[keys[0]]+ name + str(count)+'.txt', 'w') as out:
						for i in range(len(transformed_bboxes)):
							tl = (round((transformed_bboxes[i][0]-transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]-transformed_bboxes[i][3]/2)*transformed_image.shape[0]))
							br = (round((transformed_bboxes[i][0]+transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]+transformed_bboxes[i][3]/2)*transformed_image.shape[0]))
							bbox = (tl[0], tl[1], br[0], br[1])
							if checkBbox(bbox, image.shape) == 2:	
								out.write(transformed_classLabels[i] + ' ' + str(transformed_bboxes[i][0]) + ' ' + str(transformed_bboxes[i][1]) + ' ' + str(transformed_bboxes[i][2]) + ' ' + str(transformed_bboxes[i][3]) + '\n')
					out.close()


					for j in range(1, len(keys)):
						img2  = imagesPath[keys[j]] + file + '.png'

						if os.path.exists(img2):
							image2 = cv2.imread(img2)

							transformed2 = A.ReplayCompose.replay(transformed['replay'], image=image2, bboxes=bboxes, classLabels = classLabels)
							transformed_image2 = transformed2['image']
							cv2.imwrite(imagesPath[keys[j]] + name + str(count) + '.png' , transformed_image2)

							with open(labelsPath[keys[j]]+ name + str(count)+'.txt', 'w') as out:
								for i in range(len(transformed_bboxes)):
									tl = (round((transformed_bboxes[i][0]-transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]-transformed_bboxes[i][3]/2)*transformed_image.shape[0]))
									br = (round((transformed_bboxes[i][0]+transformed_bboxes[i][2]/2)*transformed_image.shape[1]), round((transformed_bboxes[i][1]+transformed_bboxes[i][3]/2)*transformed_image.shape[0]))
									bbox = (tl[0], tl[1], br[0], br[1])
									if checkBbox(bbox, image.shape) == 2:	
										out.write(transformed_classLabels[i] + ' ' + str(transformed_bboxes[i][0]) + ' ' + str(transformed_bboxes[i][1]) + ' ' + str(transformed_bboxes[i][2]) + ' ' + str(transformed_bboxes[i][3]) + '\n')
							out.close()

					count += 1
		
				#cv2.imshow('frame', display_image)
				#cv2.waitKey(0)


if __name__ == "__main__":
	main()