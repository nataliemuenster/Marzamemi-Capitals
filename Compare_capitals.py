import numpy as np
import math
import Postprocess_capitals as process
import Mesh_slices as mesh_slices
import Evaluate_similarity as similarity
import correlate
import csv

csvFile = 'capital_features.csv'
capitals = ['0262','0952','0953','0954','0955','2281','2282','2285']
characteristics = {}
features = np.zeros([len(capitals), similarity.numFeatures()])

for i,c in enumerate(capitals):
	#print c
	vertices, characteristics[c] = process.process_capital(c)
	mesh_slices.compare_slices(c, vertices, characteristics[c])
	#print "CHARACTERISTICS for capital " + c + ": " + str(characteristics[c])
	features[i] = similarity.evaluate(characteristics[c])
	#calculate normalized erosion and append to correct index in features?
	#print "FEATURES for capital " + c + ": " + str(features[i])
#print characteristics

featuresNormalized = similarity.normalize_features(features)

with open(csvFile, 'w') as file:
	writer = csv.writer(file)
	#writer.writerows([capitals])
	data = np.vstack((capitals, np.transpose(featuresNormalized)))
	writer.writerows(data)
print "Writing data to csv complete"

correlate.main()

'''with open(csvFile, 'w') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=capitals)
	writer.writeheader()
	for j in xrange(features.shape[0]): #(how many features per column)
		writer.writerow()
'''

#uncomment below
'''similarities = np.empty([len(capitals), len(capitals)])
for i,c1 in enumerate(capitals):
	for j,c2 in enumerate(capitals):
		if i != j:
			similarities[i,j] = similarity.compare(characteristics[c1], characteristics[c2])
		#else: Default value for diagonals that aren't yet set in place? (i = j)
'''