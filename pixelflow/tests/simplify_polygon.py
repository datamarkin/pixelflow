import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
from simplification.cutil import simplify_coords_vw
from shapely.geometry import Polygon
import time

# Load your data from file
file_path = './polygon.txt'
with open(file_path, 'r') as file:
    data = eval(file.read())

# Convert the flattened list to coordinate pairs (x, y)
coordinates = [(data[i], data[i+1]) for i in range(0, len(data), 2)]

# Apply RDP Algorithm
# Measure execution time for RDP Algorithm
start_rdp = time.time()
rdp_simplified = rdp(coordinates, epsilon=2.0)
end_rdp = time.time()
rdp_time = end_rdp - start_rdp

# Apply VW Algorithm
# Measure execution time for VW Algorithm
start_vw = time.time()
vw_simplified = simplify_coords_vw(np.array(coordinates), 0.01)
vw_simplified = [tuple(point) for point in vw_simplified]
end_vw = time.time()
vw_time = end_vw - start_vw

# Apply Shapely Simplification
# Measure execution time for Shapely Simplification
polygon = Polygon(coordinates)
start_shapely = time.time()
shapely_simplified = polygon.simplify(tolerance=2.0, preserve_topology=True)
shapely_simplified_coords = list(shapely_simplified.exterior.coords)
end_shapely = time.time()
shapely_time = end_shapely - start_shapely

# Plot the Original and Simplified Polygons
plt.figure(figsize=(15, 10))

# Original Polygon
plt.subplot(1, 4, 1)
plt.plot(*zip(*coordinates), marker='o', color='b', label='Original Polygon')
plt.title(f'Original Polygon ({len(coordinates)} points)')
plt.legend()

# RDP Simplified Polygon
plt.subplot(1, 4, 2)
plt.plot(*zip(*rdp_simplified), marker='o', color='g', label='RDP Simplified')
plt.title(f'RDP Simplified ({len(rdp_simplified)} points)')
plt.legend()

# VW Simplified Polygon
plt.subplot(1, 4, 3)
plt.plot(*zip(*vw_simplified), marker='o', color='r', label='VW Simplified')
plt.title(f'VW Simplified ({len(vw_simplified)} points)')
plt.legend()

# Shapely Simplified Polygon
plt.subplot(1, 4, 4)
plt.plot(*zip(*shapely_simplified_coords), marker='o', color='purple', label='Shapely Simplified')
plt.title(f'Shapely Simplified ({len(shapely_simplified_coords)} points)')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Print the number of points for all simplified polygons
print(f'RDP Simplified Points: {len(rdp_simplified)}, duration: {rdp_time}')
print(f'VW Simplified Points: {len(vw_simplified)}, duration: {vw_time}')
print(f'Shapely Simplified Points: {len(shapely_simplified_coords)}, duration: {shapely_time}')
