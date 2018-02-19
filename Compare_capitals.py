import math
import Postprocess_capitals as process
import Mesh_slices as mesh_slices

capitals = ["0955"]#, "0952", "0953", "0262"]
characteristics = {}

for c in capitals:
	characteristics[c] = process.process_capital(c)
	mesh_slices.compare_slices(c, characteristics[c])
#print characteristics