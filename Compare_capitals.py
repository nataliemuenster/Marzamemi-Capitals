import math
import Postprocess_capitals as process
import Mesh_slices as mesh_slices

capitals = ['0954','0955', '2282', "0952", "0953", "0262"]
characteristics = {}

for c in capitals:
	vertices, characteristics[c] = process.process_capital(c)
	mesh_slices.compare_slices(c, vertices, characteristics[c])
	print "Capital " + c + ": " + str(characteristics[c])
#print characteristics