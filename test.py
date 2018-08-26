import numpy as np


radius_communicate = 6
num_row = 25
num_column = 25
center_x = np.random.randint(radius_communicate, num_row-radius_communicate)
center_y = np.random.randint(radius_communicate, num_column-radius_communicate)
center = (center_x, center_y)
print(center)