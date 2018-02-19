import numpy
import math
from stl import mesh

numSlices = 5

def get_slice_intervals(height):
	#Calculate z plane locations for slices by splitting the capital into numSlices + 1 pieces and shifting up by half one of those slices (to account for not perfectly parallel ends)
	#in order from z = 0 upward, so top of capital is first
	return [i*height/float(numSlices+1) + height/(2*(numSlices+1)) for i in xrange(numSlices)]

def find_slice_area(filename):
	single_slice = mesh.Mesh.from_file(filename)
	print len(single_slice.vectors)
	print "AREAS!!"
	print single_slice.areas

def compare_slices(capitalNum, characteristics):
	sliceIntervals = get_slice_intervals(characteristics['height'])
	print "SLICE INTERVALS: " + str(sliceIntervals)
	filename = "../Mesh_slices/Lot" + capitalNum + "_10000tri_mesh_slice"
	sliceDist = sliceIntervals[1] - sliceIntervals[0]
	sliceAreas = []
	gradients = []
	avgGradient = 0
	for i in xrange(numSlices):
		sliceAreas.append(find_slice_area(filename + str(i+1) + ".stl"))
		if i > 0:
			gradient = sliceAreas[i] - sliceAreas[i-1]
			gradients.append(gradient)
			avgGradient += gradient
	avgGradient /= (numSlices-1)
	characteristics['avgGradient'] = avgGradient
