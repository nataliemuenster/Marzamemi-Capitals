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

def compute_transform_matrix(top_normal):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38

    destination_axis = np.array([0,0,-1]) #want top to face down, so the normal aligns with the -z axis
    if top_normal[0] == 0.0 and top_normal[1] == 0.0 and top_normal[2] == 1.0: 
        R = np.array([[-1,0,0],[0,1,0],[0,0,-1]]) #only happen when z angle is 180? or other angles?????????????????
    else:

        '''v = np.cross(top_normal, destination_axis)
        s = np.linalg.norm(v)
        c = np.dot(top_normal, destination_axis)
        if c == -1: return np.eye(3)

        v_x = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
            ])
        
        R = np.eye(3) + v_x + v_x*v_x * (1/(1+c))

        print str(R)
        return R'''

        uvw = np.cross(top_normal, destination_axis) #get axis -- IS THIS THE RIGHT ORDER??

        rcos = np.dot(top_normal, destination_axis)
        rsin = np.linalg.norm(uvw)

        if not np.isclose(rsin, 0):# and uvw != 0: #normalize axis
            uvw /= rsin
        u, v, w = uvw

        R = (
            rcos * np.eye(3) +
            rsin * np.array([
                [ 0, -w,  v],
                [ w,  0, -u],
                [-v,  u,  0]
            ]) +
            (1.0 - rcos) * uvw[:,None] * uvw[None,:]
        )
    #print "R: " + str(R)
    return R

#rotate so capital is lying on its top in the x-y plane at z=0
#rotate around z-axis through z centroid
def transform_capital_onto_top(capital_mesh, dims):
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
    #print top_normal
    transform_matrix = compute_transform_matrix(top_normal)
    #transform_matrix = capital_mesh.rotation_matrix([0,1,0], math.radians(180))
    transform_matrix_r = np.vstack((transform_matrix, [0,0,0]))
    transform_matrix_t = np.hstack((transform_matrix_r, [[0],[0],[0],[1]]))
    print str(transform_matrix_t)
    capital_mesh.transform(transform_matrix_t)
    
    numFaces = capital_mesh.points.shape[0]
    #print "POINT: " + str(capital_mesh.points[0:2].reshape((2,3,3)))
    #print "POINT_z: " + str(np.amin(capital_mesh.points[0:2].reshape((2,3,3)), axis=(0,1))[2])
    move_z = np.amin(capital_mesh.points.reshape((numFaces,3,3)), axis=(0,1))[2] #get the min z; each row contains the 3 3D points
    print move_z
    translation = np.array([[0],[0],[0-move_z],[1]])
    transform_matrix = np.vstack((np.eye(3), [0,0,0]))
    transform_matrix = np.hstack((transform_matrix, translation))
    print str(transform_matrix)
    capital_mesh.transform(transform_matrix)
    
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
print "THIRD VERTEX!: " + str(vertices_np[3])
'''Make use of numpy-stl mesh library'''
#create the mesh
capital_mesh = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))

for i, f in enumerate(faces_np):
    for j in range(3):
        capital_mesh.vectors[i][j] = vertices_np[f[j]-1,:]
# Write the mesh to file
capital_mesh.save(MESHFILENAME)
print "cap mesh 3: " + str(capital_mesh[3])
#print capital_mesh.normals
dims = get_capital_dimensions(capital_mesh)
transform_capital_onto_top(capital_mesh, dims)

create_plot(capital_mesh)

