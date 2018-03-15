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
features = np.zeros([len(capitals), len(similarity.getFeatureLabels())])

for i,c in enumerate(capitals):
	vertices, characteristics[c] = process.process_capital(c)
	mesh_slices.compare_slices(c, vertices, characteristics[c])
	features[i] = similarity.evaluate(characteristics[c])

#similarity.plotFeatures(capitals, features)

#PCA
featuresNormalizedPCA = similarity.preprocess_for_PCA(features)
featuresPCA = similarity.PCA(featuresNormalizedPCA)
similarity.plotPCA(capitals,featuresPCA)
quit()
featuresNormalized = similarity.normalizeAllFeatures(features)

with open(csvFile, 'w') as file:
	writer = csv.writer(file)
	data = np.vstack((capitals, np.transpose(featuresNormalized)))
	writer.writerows(data)
print "Writing data to csv complete"

correlate.main()




#uncomment below
'''similarities = np.empty([len(capitals), len(capitals)])
for i,c1 in enumerate(capitals):
	for j,c2 in enumerate(capitals):
		if i != j:
			similarities[i,j] = similarity.compare(characteristics[c1], characteristics[c2])
		#else: Default value for diagonals that aren't yet set in place? (i = j)
'''