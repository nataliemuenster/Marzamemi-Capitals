import math
import Postprocess_capitals as process

capitals = ["0262", "0952", "0953", "0955"]
characteristics = {}

for c in capitals:
	characteristics[c] = process.processCapital(c)

print characteristics