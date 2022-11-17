from jakteristics import las_utils, compute_features, FEATURE_NAMES
import os
from tkinter import filedialog
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pickle
import math

print(FEATURE_NAMES)
#FEATURE_NAMES = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']


folder = filedialog.askdirectory(title = "Select the folder with the LAS files")

trainVal = []

for f in os.listdir(folder):
	trainVal.append(folder + '/' + f + '/')

meanTest = {}
stdTest = {}
for i in FEATURE_NAMES:
	meanTest[i] = []
	stdTest[i] = []

X = []
for f in trainVal:
	for las in os.listdir(f):
		xyz = las_utils.read_las_xyz(f + '/' + las)
		#print(np.max(xyz, axis=0)-np.min(xyz, axis=0))
		#print(np.max(xyz, axis=0))
		#print(np.min(xyz, axis=0))
		#print('------')


		features = compute_features(xyz, search_radius=3)#, feature_names = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality'])
		
		if np.isnan(features).any() == False:
			stats = {}
			for i in FEATURE_NAMES:
				stats[i] = []
			
			for feature in features:
				for i in range(len(FEATURE_NAMES)):
					stats[FEATURE_NAMES[i]].append(feature[i])

			tmp = []

			for i in FEATURE_NAMES:		
				mean = np.mean(stats[i])
				stdev = np.std(stats[i])
				tmp += [mean,stdev]
				meanTest[i].append(mean)
				stdTest[i].append(stdev)


			#tmp += list(np.max(xyz, axis=0)-np.min(xyz, axis=0))


			X.append(tmp)


#print(X)

for i in meanTest:
	print(i + ': ' + str(max(meanTest[i]))+ ' - ' + str(max(stdTest[i])))

#clf = OneClassSVM(verbose = 1, nu=0.1, gamma=0.1).fit(X)
#clf = linear_model.SGDOneClassSVM(random_state=2, verbose = 1).fit(X)
clf = LocalOutlierFactor(n_neighbors=1, novelty=True).fit(X)
#clf.fit(X)
predictions = clf.predict(X)

#for i in range(len(predictions)):
	#print(str(X[i]) + ': ' + str(predictions[i]))
	

print(sum(predictions==1))
print(len(predictions))
pickle.dump(clf, open('test.sav', 'wb'))

