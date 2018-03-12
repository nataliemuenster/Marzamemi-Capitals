import csv
import random
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

MIN_CORRELATION = -1
DEFAULT = 3
FILE_NAME = 'random.csv'

# This program finds the correlation between all the 
# capitals in the given csv file. It puts the correlatios
# into a correlation matrix, where similar items are sorted
# to be close to one another.
def main():
	# load data
	capitals, capitalMap = loadData(FILE_NAME)
	n = len(capitals)

	# find all correlations
	maxCorrelation = MIN_CORRELATION
	argMax = None
	for i in range(n):
		for j in range(n):
			if i == j: continue
			x = capitalMap[capitals[i]]
			y = capitalMap[capitals[j]]
			corr = calculateCorrelation(x, y)
			if corr > maxCorrelation:
				maxCorrelation = corr
				argMax = (i, j)
	print argMax, maxCorrelation

# Return the Pearson correlation between two different random variables (based
# on equally weighted samples)
def calculateCorrelation(x, y):
	return stats.pearsonr(x, y)[0]

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
			value = int(row[i])
			capitalMap[headers[i]].append(value)
	return headers, capitalMap


if __name__ == '__main__':
	main()