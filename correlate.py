'''Cite: Chris Piech's CS109 code'''

import csv
import random
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import Evaluate_similarity as evaluate

MIN_CORRELATION = -1
DEFAULT = 3
FILE_NAME = 'capital_features.csv'
SORT_MATRIX = True

# This program finds the correlation between all the 
# capitals in the given csv file. It puts the correlatios
# into a correlation matrix, where similar items are sorted
# to be close to one another.
def main():
	# load data
	capitals, capitalMap = loadData(FILE_NAME)
	n = len(capitals)

	# find all correlations
	correlationMatrix = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			x = capitalMap[capitals[i]]
			y = capitalMap[capitals[j]]
			#corr = calculateCorrelation(x, y)
			corr = evaluate.calculateSimilarity(np.array(x),np.array(y))
			correlationMatrix[i][j] = corr
		#quit()

	# sort and display
	if SORT_MATRIX:
		sortedIndices, capitals = sortcapitals(correlationMatrix, capitals)
		correlationMatrix = sortMatrix(sortedIndices, correlationMatrix)
	normalizeDiagonal(correlationMatrix)
	makeFigure(correlationMatrix, capitals)

# Return the Pearson correlation between two different random variables (based
# on equally weighted samples)
def calculateCorrelation(x, y):
	return stats.pearsonr(x, y)[0]

# takes in a matrix, and a desired ordering of rows 
# and resorts the matrix such that rows are in the desired order
def sortMatrix(sortedIndices, correlationMatrix):
	n = len(sortedIndices)
	newMatrix = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			lookupI = sortedIndices[i]
			lookupJ = sortedIndices[j]
			value = correlationMatrix[lookupI][lookupJ]
			newMatrix[i][j] = value
	return newMatrix

def normalizeDiagonal(correlationMatrix):
	n = correlationMatrix.shape[0]
	
	# Make the diagonal equal to the min element so that
	# it doesn't dominate the max.
	for i in range(n):
		correlationMatrix[i][i] = MIN_CORRELATION
	
	# Make the diagonal equal to the max element so that
	# it doesn't dominate the vizualization.
	for i in range(n):
		correlationMatrix[i][i] = max(correlationMatrix.flatten())
		
def sortcapitals(correlationMatrix, capitals):
	n = len(capitals)
	# Work on a copy of the matrix
	copyOfMatrix = np.copy(correlationMatrix)
	for i in range(len(capitals)):
		# Ignore the diagonal
		copyOfMatrix[i][i] = MIN_CORRELATION

	# Chose the first two to have the highest correlation
	indexMax = copyOfMatrix.argmax()
	first = indexMax / len(capitals)
	second = indexMax % len(capitals)
	sortedIndices = [first,second]

	# Greedily chose the next rows based to have the max
	# correlation to the previous row.
	pre = first
	curr = second
	for i in range(n - 2):
		# remove col from consideration
		for i in range(n):
			copyOfMatrix[i][pre] = MIN_CORRELATION
		# chose the next max
		nextIndex = copyOfMatrix[curr].argmax()
		sortedIndices.append(nextIndex)
		pre = curr
		curr = nextIndex

	# Also sort the textual names
	sortedNames = []
	for v in sortedIndices:
		sortedNames.append(capitals[v])
	return sortedIndices, sortedNames

def loadData(fileName):
	capitalMap = {}
	reader = csv.reader(open(fileName, 'rU'))

	# get a list of the csv headers
	headers = reader.next()
	for i in range(len(headers)):
		headers[i] = headers[i].strip()
		capitalMap[headers[i]] = []

	# read the rest of the file
	for row in reader:
		for i in range(len(headers)):
			# there are a small number of missing data. 
			# assume missing at random	
			if row[i] == '':
				row[i] = DEFAULT
			value = float(row[i])
			capitalMap[headers[i]].append(value)
	return headers, capitalMap
	
# Oh matplot lib. You can be so hard to work with :'(
def makeFigure(data, labels):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(data)
	fig.colorbar(cax)
	ax.set_xticklabels(['']+labels)
	ax.set_yticklabels(['']+labels)
	for tick in ax.get_xticklabels():
		tick.set_rotation(90)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	plt.show()

if __name__ == '__main__':
	main()