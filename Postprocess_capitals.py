import numpy as np
import stl
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math

CAPITALNUM = "0952"
FILENAME = "../Data/Lot" + CAPITALNUM + "_10000tri.obj"
MESHFILENAME = "../Data/Lot" + CAPITALNUM + "_10000tri_mesh.stl"
SKIPLINES = 8



def get_capital_dimensions(vertices): #,capital_mesh):
    minx = maxx = miny = maxy = minz = maxz = None

    vertex_min = np.amin(vertices_np, axis=0) #[minx, miny, minz]
    vertex_max = np.amax(vertices_np, axis=0) #[maxx, maxy, maxz]
    volume, cog, inertia = capital_mesh.get_mass_properties()
    return {"volume": volume, "cog": cog, "intertia": inertia, "minx": vertex_min[0], "maxx": vertex_max[0], "miny": vertex_min[1], "maxy": vertex_max[1], "minz": vertex_min[2], "maxz": vertex_max[2]}

    '''for p in capital_mesh.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    '''
    #return {"minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy, "minz": minz, "maxz": maxz}


#rotate so capital is lying on its top in the x-y plane at z=0
#rotate around z-axis through z centroid
def rotate_capital_onto_top(capital_mesh, dims):
    #identify top plane using bounding box dimensions -- the top/bottom face is likely the one with the most equivalent length/width ratio
    top_face = None
    #diff_x = math.fabs(dims['maxx'] - dims['minx'])
    #diff_y = math.fabs(dims['miny'] - dims['maxy'])
    #diff_z = math.fabs(dims['miny'] - dims['maxy'])
    #smallest_diff = float('inf')
    #if math.fabs(diff_x - diff_y) < smallest_diff:
    #	smallest_diff = math.fabs(diff_x - diff_y)
    #	top_face = "XY"
    #if math.fabs(diff_x - diff_z) < smallest_diff:
    #	smallest_diff = math.fabs(diff_x - diff_z)
    #	top_face = "XZ"
    #if math.fabs(diff_y - diff_z) < smallest_diff:
    #	smallest_diff = math.fabs(diff_y - diff_z)
    #	top_face = "YZ"
    #print "TOP FACE: " + top_face

    #z_center = dims["maxz"] - dims["minz"]
    #z_axis = [0.0, 0.0, 100] #WHY 100?
    #theta = math.radians(180)
    #transformation_matrix = np.array([[1,0,0,0],[0, math.cos(theta), -math.sin(theta), 0],[0, math.sin(theta), math.cos(theta), 0],[0,0,0,1]])

    transform_matrix = None
    if CAPITALNUM == "0952":
        #find surface normal of top plane: upper top face is on XY plane -- all surface points are at max z
        face_pts = np.array([[dims["minx"],dims["miny"],dims["maxz"]], [dims["maxx"],dims["miny"],dims["maxz"]], [dims["maxx"],dims["maxy"],dims["maxz"]]])
        top_normal = np.cross(face_pts[1] - face_pts[0], face_pts[1] - face_pts[2])
        points_mid = np.array((face_pts[2] + face_pts[1] + face_pts[0]) / 3.0)
    
    #Scalar product: make sure the top surface normal is pointing away from the capital center
    top_normal /= np.linalg.norm(top_normal)
    if np.dot(points_mid - dims["cog"], top_normal) < 0: #center of gravity here should really be centroid, but they should be similar for these objects
        top_normal *= -1
    print "TOP NORMAL: " + str(top_normal)

    #transform_matrix = np.array([[1,0,0,0],[0, math.cos(math.radians(180)), -math.sin(math.radians(180)), 0],[0, math.sin(math.radians(180)), math.cos(math.radians(180)), 0],[0,0,0,1]])

    #capital_on_top = capital_mesh.transform(Lot0952_transform_matrix)
    #capital_mesh.rotate(z_axis, math.radians(90))

def create_plot(capital_mesh):
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

    #def cut_slice(vertices, faces):
    #slice = meshcut.cross_section(vertices, faces, plane_orig=(1.2, -0.125, 0), plane_normal=(1, 0, 0))

vertices = [] #expect v 1 2 3
faces = [] #expect f 1 2 3
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
dims = get_capital_dimensions(capital_mesh)
rotate_capital_onto_top(capital_mesh, dims)

create_plot(capital_mesh)

