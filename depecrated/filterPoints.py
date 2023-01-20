import laspy
import os

i = 'unfilteredLAS/'
o = "LAS/"

for file in os.listdir(i):
	print(file)
	las = laspy.read(i+file)

	new_file = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
	new_file.points = las.points[las.classification == 2]

	new_file.write(o+file)