import numpy as np
import math
import Postprocess_capitals as process
import Mesh_slices as mesh_slices
import Evaluate_similarity as similarity
import csv

csvFile = 'features.csv'
capitals = ['0954','0955', '2282', "0952", "0953", "0262"]
characteristics = {}
features = [] #use numpy array instead?

for i,c in enumerate(capitals):
	vertices, characteristics[c] = process.process_capital(c)
	mesh_slices.compare_slices(c, vertices, characteristics[c])
	print "CHARACTERISTICS for capital " + c + ": " + str(characteristics[c])
	features.append(similarity.evaluate(characteristics[c]))
	print "FEATURES for capital " + c + ": " + str(features[i])
#print characteristics

'''with open(csvFile, 'w') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=capitals)
	writer.writeheader()
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