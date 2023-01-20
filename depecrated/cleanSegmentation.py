import csv
import sys
csv.field_size_limit(sys.maxsize)

tmp = "Images/LRM/tmp.csv"
outFile = "Images/LRM/Segmentation.csv"
classes = ["2.1.1.1", "2.1.1.2", "2.2.1.1", "2.2.2.1", "2.2.3.1", "2.3.1.1", "2.3.1.2", "2.3.1.3", "2.3.3.1", "2.4.1.1", "3.1.1.1", "3.1.2.1", "4.1.1.1", "4.1.1.2", "4.1.1.3", "4.1.1.4", "4.1.1.5", "4.1.1.6", "4.1.1.7", "5.1.1.1", "5.1.1.2", "5.1.1.3", "5.1.1.4", "5.1.1.5", "5.1.1.6", "5.1.1.7", "5.1.2.1", "5.1.2.2", "5.1.2.3", "6.1.1.1", "7.1.3.1"]

with open(tmp) as i, open(outFile, 'w') as o:
	reader = csv.DictReader(i)
	writer = csv.writer(o)
	writer.writerow(["WKT", "COS18n4_C"])
	for row in reader:
		if row["COS18n4_C"] in classes:
			writer.writerow([row["WKT"], row["COS18n4_C"]])