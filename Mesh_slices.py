import numpy as np
import math
from stl import mesh

numSlices = 5
#SKIPLINES = 4
ZRange = 3 #how many units the slice can span on either side of interval value

def get_slice_intervals(vertices, height):
	#Calculate z plane locations for slices by splitting the capital into numSlices + 1 pieces and shifting up by half one of those slices (to account for not perfectly parallel ends)
	#in order from z = 0 upward, so top of capital is first
	#return [i*height/float(numSlices+1) + height/(2*(numSlices+1)) for i in xrange(numSlices)] #First method for determining intervals:
	#evenly spaced along capital, starting at points on either end that are an arbitrary short distance away from the ends
	

	#second method for determining intervals: start with location of widest point of capital top as the initial slice, then evenly spaced until end of capital
	lowerThird = vertices[np.where(vertices[:,2] < height/3)[0]]
	minxy = np.argmin(lowerThird, axis=0)[:2]
	maxxy = np.argmax(lowerThird, axis=0)[:2]
	z_midX = (lowerThird[maxxy[0]][2] + lowerThird[minxy[0]][2]) / 2.0 #find midpoint of Zs where max and min Xs are
	z_midY = (lowerThird[maxxy[1]][2] + lowerThird[minxy[1]][2]) / 2.0 #find midpoint of Zs where max and min Ys are
	z_maxWidth = (z_midX + z_midY) / 2.0 #find midpoint of Zs where X is widest and Y is widest
	sliceDist = (height - z_maxWidth) / numSlices
	return [z_maxWidth + i*sliceDist for i in xrange(numSlices)] #first slice is at widest part of top, last slice is an interval away from end to account for slanted end

def find_slice_dimensions(interval, vertices):
	#singleSlice = process_slice(capitalNum) #First attempt
	#Third attempt:
	#singleSlice = find_rectangle_approximation(vertices)
	#find all vertices with z-value near the slice location
	sliceIndices = np.where(np.logical_and(vertices[:,2] >= interval-ZRange, vertices[:,2] <= interval+ZRange))[0]
	sliceVertices = [vertices[i] for i in sliceIndices]
	x_min, y_min = np.amin(sliceVertices, axis=0)[:2] #[minx, miny]
   	x_max, y_max = np.amax(sliceVertices, axis=0)[:2] #[maxx, maxy]
   	maxRectArea = math.fabs((x_max-x_min) * (y_max-y_min)) #will this ever be negative?
	return [x_min, x_max, y_min, y_max, maxRectArea]

def compare_slices(capitalNum, vertices, characteristics):
	sliceIntervals = get_slice_intervals(vertices, characteristics['height'])
	#print "SLICE INTERVALS: " + str(sliceIntervals)
	sliceDist = sliceIntervals[1] - sliceIntervals[0]
	
	gradients = []
	avgGradient = 0
	#totalGradient = 0
	sliceDims = [] #each element is a list of the slice dimenesions
	sideEdges = [[],[],[],[]] #capital has four side edges along z axis -- store location of corners for each slice
	edgeSlopes = [[],[],[],[]] #instantiate to empty arrays

	for interval in sliceIntervals:
		sliceDims.append(find_slice_dimensions(interval, vertices))

	for i in xrange(numSlices):
		#areas
		if i > 0:
			gradient = (sliceDims[i][-1] - sliceDims[i-1][-1]) / sliceDist
			gradients.append(gradient)
			#evaluate how much the gradients differ to measure consistency of slope? erosion
			avgGradient += gradient
			#totalGradient += sliceDims[i][-1]

		#edge slopes (add each corner of the slice)
		#do i need to store these?
		sideEdges[0].append((sliceDims[i][0], sliceDims[i][2])) #minx, miny
		sideEdges[1].append((sliceDims[i][0], sliceDims[i][3])) #minx, maxy
		sideEdges[2].append((sliceDims[i][1], sliceDims[i][2])) #maxx, miny
		sideEdges[3].append((sliceDims[i][1], sliceDims[i][3])) #maxx, maxy
		
		if i > 0:
			edgeSlopes[0].append((sliceDims[i][0] - sliceDims[i-1][0], sliceDims[i][2] - sliceDims[i-1][2]))
			edgeSlopes[1].append((sliceDims[i][0] - sliceDims[i-1][0], sliceDims[i][3] - sliceDims[i-1][3]))
			edgeSlopes[2].append((sliceDims[i][1] - sliceDims[i-1][1], sliceDims[i][2] - sliceDims[i-1][2]))
			edgeSlopes[3].append((sliceDims[i][1] - sliceDims[i-1][1], sliceDims[i][3] - sliceDims[i-1][3]))
		#evaluate how much the gradients differ to measure roughness?

	#slope between start and end slice, per corner:
	sideSlopes = [((sideEdges[0][0][0] - sideEdges[0][-1][0]) / (sliceDist * numSlices), (sideEdges[0][0][1] - sideEdges[0][-1][1]) / (sliceDist * numSlices)),
					((sideEdges[1][0][0] - sideEdges[1][-1][0]) / (sliceDist * numSlices), (sideEdges[1][0][1] - sideEdges[1][-1][1]) / (sliceDist * numSlices)),
					((sideEdges[2][0][0] - sideEdges[2][-1][0]) / (sliceDist * numSlices), (sideEdges[2][0][1] - sideEdges[2][-1][1]) / (sliceDist * numSlices)),
					((sideEdges[3][0][0] - sideEdges[3][-1][0]) / (sliceDist * numSlices), (sideEdges[3][0][1] - sideEdges[3][-1][1]) / (sliceDist * numSlices))]
	# ^ Make the signs correct for direction
	print "SIDE EDGE SLOPES: " + str(sideSlopes)
	characteristics['side edge slopes'] = sideSlopes
	#totalGradient /= (sliceDist * numSlices)
	#print "SLICEDIST: " + str(sliceDist) + ", TOTAL AREA GRADIENT: " + str(totalGradient)
	#if I want top dimensions and not just area, restructure get_slice_interval briefly
	avgGradient /= (numSlices)
	characteristics['top_area'] = sliceDims[0][-1]
	
	characteristics['slice areas'] = np.array(sliceDims)[:,-1]
	print "RECTANGLES SLICES: " + str(characteristics['slice areas'])

	characteristics['avg area gradient'] = avgGradient



'''#First attempt at finding slice area -- parsing a .obj file of vertices + traingles of single plane 
#to then try to get outised edges to use shoelace formula for area
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

    #Make use of numpy-stl mesh library
    #create the mesh
    meshSlice = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces_np):
        for j in range(3):
            meshSlice.vectors[i][j] = vertices_np[f[j]-1,:]

    return meshSlice
'''