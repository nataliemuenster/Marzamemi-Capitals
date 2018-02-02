import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

FILENAME = "Lot0952_50tri.obj"
MESHFILENAME = "Lot0952_50tri_mesh.stl"
SKIPLINES = 8


vertices = [] #expect v 1 2 3
faces = [] #expect f 1 2 3
minmax_x = {"min": float('inf'), "max":  -float('inf')}
minmax_y = {"min": float('inf'), "max":  -float('inf')}
minmax_z = {"min": float('inf'), "max":  -float('inf')}
lineNum = 0
for line in open(FILENAME):
	if lineNum < SKIPLINES:
		lineNum += 1
		continue
	data = line.rstrip('\n').split()
	#print data
	if len(data) <= 0: 
		continue
	if data[0] == 'v':
		data_vals = [float(d) for d in data[1:]]
		vertices.append(data_vals) #should append a vector of ints representing vertex indices
		if data_vals[0] > minmax_x["max"]: minmax_x["max"] = data_vals[0]
		if data_vals[0] < minmax_x["min"]: minmax_x["min"] = data_vals[0]
		if data_vals[1] > minmax_y["max"]: minmax_y["max"] = data_vals[1]
		if data_vals[1] < minmax_y["min"]: minmax_y["min"] = data_vals[1]
		if data_vals[2] > minmax_z["max"]: minmax_z["max"] = data_vals[2]
		if data_vals[2] < minmax_z["min"]: minmax_z["min"] = data_vals[2]

	elif data[0] == 'f':
		faces.append([int(i) for i in data[1:]]) #should append a vector
if lineNum <= 1:
	print "Error, missing elements in file."
	quit()

vertices_np = np.array(vertices)
faces_np = np.array(faces)

'''Make use of numpy-stl mesh library'''
#create the mesh
capital_mesh = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces_np):
    for j in range(3):
        capital_mesh.vectors[i][j] = vertices_np[f[j]-1,:]
# Write the mesh to file
capital_mesh.save(MESHFILENAME)
#print capital_mesh.normals

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
# Load the STL files and add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(capital_mesh.vectors))
# Auto scale to the mesh size
scale = capital_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)
# Show the plot to the screen
pyplot.show()






'''print type(minmax_x["max"])
print minmax_x

#Find rough object size to determine sides
half_x = (minmax_x["max"] - minmax_x["min"]) / 2.0
half_y = (minmax_y["max"] - minmax_y["min"]) / 2.0
half_z = (minmax_z["max"] - minmax_z["min"]) / 2.0

total_normal = 0

#find surface normals:
for f in faces:
	#for each triangle, find its surface normal vector with the cross vector of its vertices
	f_vertices = [vertices_np[f[0]-1], vertices_np[f[1]-1], vertices_np[f[2]]-1] #account for 0-indexing
	#size of normal is function of face's area -- don't normalize bc this tells us how significant this piece is in the capital's geometry
	f_normal = np.cross(f_vertices[i+1]-f_vertices[i], f_vertices[i+2]-f_vertices[i])
	total_normal += f_normal
print "TOTAL NORMAL:" + str(total_normal)
	#build a bounding box?
	#determine id point is inside or outside mesh?
'''