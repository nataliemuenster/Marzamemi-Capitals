import numpy as np
import math
from stl import mesh

numSlices = 5
SKIPLINES = 4

def get_slice_intervals(height):
	#Calculate z plane locations for slices by splitting the capital into numSlices + 1 pieces and shifting up by half one of those slices (to account for not perfectly parallel ends)
	#in order from z = 0 upward, so top of capital is first
	return [i*height/float(numSlices+1) + height/(2*(numSlices+1)) for i in xrange(numSlices)]

def find_slice_area(capitalNum):
	#singleSlice = process_slice(capitalNum) #First attempt
	#Third attempt:
	filename = "..Data/Lot"
	singleSlice = find_rectangle_approximation(filename)
	print len(singleSlice.normals)
	print "AREAS!!"
	print singleSlice.areas

def compare_slices(capitalNum, characteristics):
	sliceIntervals = get_slice_intervals(characteristics['height'])
	print "SLICE INTERVALS: " + str(sliceIntervals)
	
	sliceDist = sliceIntervals[1] - sliceIntervals[0]
	sliceAreas = []
	gradients = []
	avgGradient = 0
	for i in xrange(numSlices):
		sliceAreas.append(find_slice_area(filename + str(i+1) + ".obj"))
		if i > 0:
			gradient = sliceAreas[i] - sliceAreas[i-1]
			gradients.append(gradient)
			avgGradient += gradient
	avgGradient /= (numSlices-1)
	characteristics['avgGradient'] = avgGradient

'''First attempt at finding slice area -- parsing a .obj file of vertices + traingles of single plane 
to then try to get outised edges to use shoelace formula for area'''
def process_slice(capitalNum):
    filename = "../Mesh_slices/Lot" + capitalNum + "_10000tri_mesh_slice"
    vertices = [] #expect v 1 2 3
    faces = [] #expect f 1 2 3
    lineNum = 0
    firstRound = 1 #some files will have multiple sets of data -- only take first one
    for line in open(filename):
        if lineNum < SKIPLINES:
            lineNum += 1
            continue
        data = line.rstrip('\n').split()
        #print data
        if len(data) <= 0: 
            continue

        if data[0] == 'v':
            if firstRound == 0: break #v's will come after f's only if onto second set of data
            data_vals = [float(d) for d in data[1:]]
            vertices.append(data_vals) #should append a vector of ints representing vertex indices
            
        elif data[0] == 'f':
            firstRound = 0
            #should append a vector -- this format has 1/1/1 insead of 1, so this gets rid of trailing slashes and copies
            print data
            facePts = [(int(data[1][:(len(data[1]) - 2) / 3])), (int(data[2][:(len(data[2]) - 2) / 3])), (int(data[3][:(len(data[3]) - 2) / 3]))]
            faces.append(facePts)

    if lineNum <= 1:
        print "Error, missing elements in file."
        quit()

    vertices_np = np.array(vertices)
    faces_np = np.array(faces)

    '''Make use of numpy-stl mesh library'''
    #create the mesh
    meshSlice = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces_np):
        for j in range(3):
            meshSlice.vectors[i][j] = vertices_np[f[j]-1,:]

    return meshSlice