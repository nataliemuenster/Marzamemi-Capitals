import math
import numpy as np
import matplotlib.pyplot as plt

featureLabels = ['varianceArea', 'varianceCenter', 'sideDiff', 'erosion', 'avg slice area', 'avg area gradient', 'height', 'volume'] #just for reference
erosionWeights = np.array([100, 0.005, 0.05])
similarityWeights = np.array([1])
#erosionRange = 2.0/3

def getFeatureLabels():
	return featureLabels

#how to evaluate -- which ones are most similar on a relative scale, or on a definite scale where they get their own uncorrelated score?
#establish weights and values for erosion, volume, proportion (height vs. lenghts/width), side slope
#kmeans?

def plotFeatures(capitals, features):
	plt.figure(1)

	height_idx = featureLabels.index('height')
	ax = plt.subplot(111)
	plt.bar(np.arange(0, len(capitals)), features[:,height_idx])
	xlabels = np.hstack((['0000'],capitals))
	ax.set_xticklabels(xlabels)
	ax.set_xlabel('Capital')
	ax.set_ylabel('Height (unit?)')
	ax.set_title('Height')
	plt.show()
	
	volume_idx = featureLabels.index('volume')
	ax = plt.subplot(111)
	plt.bar(np.arange(0, len(capitals)), features[:,volume_idx])
	xlabels = np.hstack((['0000'],capitals))
	ax.set_xticklabels(xlabels)
	ax.set_xlabel('Capital')
	ax.set_ylabel('Volume (unit?)') #(e8)
	ax.set_title('Volume')
	plt.show()

	erosion_idx = featureLabels.index('erosion')
	erosionNormalized = normalizeSingleFeature(features[:, erosion_idx]) #only normalize erosion here

	ax = plt.subplot(111)
	plt.bar(np.arange(0, len(capitals)), erosionNormalized)
	xlabels = np.hstack((['0000'],capitals))
	ax.set_xticklabels(xlabels)
	ax.set_xlabel('Capital')
	ax.set_title('Erosion')

	plt.show()


#returns an array of weighted values for a single capital
def evaluate(capital):
	capitalValues = np.zeros([2])
	features = []
	#evaluate erosion by how asymmetric it is with respect to the centerline

	#evaluate erosion or sloped side design by how much the gradient bw areas of slices differs wrt the average gradient
	gradA = capital['area gradients']
	gradC = capital['center gradients']
	varianceArea = 0.0
	varianceCenter = 0.0
	for i in xrange(len(gradA)):
		#variance = expectation of squared deviation from mean
		varianceArea += (gradA[i] - capital['avg area gradient'])**2
		varianceCenter += (gradC[i] - capital['avg center gradient'])**2
	varianceArea = (varianceArea / len(gradA))**0.5
	varianceCenter = (varianceCenter / len(gradC))**0.5
	features.append(varianceArea) #std dev. = math.fabs(varianceA**0.5)
	#center gradients is 2 dimensionsl, for x and y -- solve by finding hypotenuse for one number for variance (?????????????????)
	features.append((varianceCenter[0]**2 + varianceCenter[1]**2)**0.5)
	#print "VARIANCE AREA: " + str(varianceArea)
	#print "VARIANCE CENTER: " + str(varianceCenter)
	#claculate eq of line that defines the axis of symmetry throught the capital
	#(x,y) = capital['top center'] --> eq for perfect

	#difference bw average length vs width -- should be symmetric
	sideDiff = capital['avg side lengths diff'] #math.fabs(capital['avg slice side lengths'][0] - capital['avg slice side lengths'][1])
	features.append(sideDiff) #normalize over area or volume?
	#print "SIDE DIFF: " + str(sideDiff)

	erosion = find_erosion(varianceArea, varianceCenter, sideDiff)
	features.append(erosion) #add this intsead after all erosions found and can normalize?

	features.append(capital['avg slice area'])
	features.append(capital['avg area gradient'])
	features.append(capital['height'])
	features.append(capital['volume'])
	#print features
	return features

def find_erosion(varianceArea, varianceCenter, sideDiff):
	#erosionFeatures = np.empty([len(erosionWeights)])
	offCenter = (varianceCenter[0] + varianceCenter[1]) / 2.0 #avg or sqrt the sqs?
	#sideDiff --> unevenSides
	erosionFeatures = np.array([offCenter, varianceArea, sideDiff])
	erosion = np.dot(erosionFeatures, erosionWeights)
	return erosion

def preprocess_for_PCA(features):
	mu = 1.0/features.shape[0] * np.sum(features, axis=0) #1/number of capitals * sum of each feature over all capitals
	features_1 = features - mu
	sigma_sq = np.empty([features.shape[1]])
	for j1 in xrange(features.shape[1]): #(of attribute)
		sigma_sq[j1] = (1.0/features.shape[0]) * np.sum(features[:,j1], axis=0)**2
	for j2 in xrange(features.shape[1]): 
		features_1[:,j2] = features_1[:,j2] / sigma_sq**0.5 #just sigma, not sigma squared, right?
	return features_1

#vector of most variance in high-dimensional space -- reduce to k=2, plot them, see if there's clusters

#generative model -- output of python model = all decisions made in process, and 3d column
#generate a bunch of type A, a bunch of type B, erode them as i see fit, then see what is still similar/diff
#minimum viable generative model: choose obvious differences, simulate erosion, backtrack latent variables
#numerical techniques are going to fall to describing differences in erosion; use joint distrib bw erosion and latent variable (look past erosion)

#small data -- speaks to field of archaeology, dont realy know if youre right

#Principal Component Analysis:
#To change the number of dimensions to reduce to, alter "eigVects = eigVects[:, :__]"
def PCA(features):
	covarMatrix = features*np.transpose(features)
	eigVals, eigVects = np.linalg.eig(covarMatrix)
	#sort eigVects (decreasing order)
	idx = np.argsort(eigVals)[::-1]
	eigVects = eigVects[:,idx]
	eigVals = eigVals[idx] #sort according to same index

	eigVects = eigVects[:, :2] #we want to rescale to one dimension --> select the first __ eigenvector(s)
	x,y,z = np.dot(eigVects.T, features.T).T, eigVals, eigVects
	print x
	return x
	#capitalNum, all eigVects, eigVect^T * capital features
	#lowDimension = np.empty([eigVects.shape[0], eigVects.shape[0], eigVects.shape[0], eigVects.shape[0]]) #all same, doesn't matter
	#for i,e in enumerate(eigVects):
	#	lowDimension[i] = np.transpose(e) * features[i]
	#print lowDimension

def plotPCA(capitals, features):
	plt.figure(1)
	plt.scatter(features[:,0], features[:,1])
	#for label, x, y in zip(capitals, features[:,0], features[:,1]):
	#	plt.annotate(label,xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
	plt.xlabel('PCA1')
	plt.ylabel('PCA2')
	plt.title('Principal Component Analysis (PCA) to 2 dimensions')
	plt.show()


def normalizeSingleFeature(feature):
	featureNormalized = np.empty_like(feature)
	i_min = np.amin(feature)
	i_max = np.amax(feature)
	for i in xrange(feature.shape[0]):
		#featureNormalized[i] = (feature[i] - i_min) / (i_max - i_min)
		featureNormalized[i] = (feature[i] - 0) / (i_max - 0)
	return featureNormalized


#Normalize features so they are all measured on the same scale and do not dominate based on the values of each feature
#Either normalize with min = 0 or min = min
def normalizeAllFeatures(features):
	featuresNormalized = np.empty_like(features)
	for j in xrange(features.shape[1]):
		j_min = np.amin(features[:,j])
		j_max = np.amax(features[:,j])
		for i in xrange(features.shape[0]):
			#featuresNormalized[i][j] = (features[i][j] - j_min) / (j_max - j_min)
			featuresNormalized[i][j] = (features[i][j] - 0) / (j_max - 0)
			#if j == 1: print "features: " + str(features[i][j]) + " --> normed: " + str(featuresNormalized[i][j])
	return featuresNormalized

#similar levels of erosion -> more equivalent comparison
def calculateSimilarity(cap1, cap2):
	weights = []
	similarity = 0
	cosSimilarity = 0

	similarity = np.linalg.norm(cap2-cap1)
	cosSimilarity = np.dot(cap1,cap2) / (np.linalg.norm(cap1) * np.linalg.norm(cap2))
	#for f in xrange(len(cap1)):
		#use +/- euclidian dist
		#similarity += (cap1[f]**2 + cap2[f]**2)**0.5
		
	return similarity
	#capital[3] = erosion in features above
	#if 1-erosionRange <= cap1[3] / cap2[3] <= 1+erosionRange :
		#don't normalize, only evaluate erosion in relative measures



	'''if cap1.erosion > __ and cap2.erosion > __:
		weights = []
	elif cap1.erosion > __ and cap2.erosion < __:
		weights = []
	elif cap1.erosion < __ and cap2.erosion > __:
		weights = []

	#OR:
	if cap1.erosion - cap2.erosion > __:
		weights = 
	'''
	



