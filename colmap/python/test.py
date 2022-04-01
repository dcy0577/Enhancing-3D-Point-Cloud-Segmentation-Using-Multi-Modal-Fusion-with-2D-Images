from utils.ply import read_ply
import numpy as np

data = read_ply('/home/dchangyu/MV-KPConv/colmap/pointcloud/office_block_3 _pairing_bin.ply')
#float 32
points = np.vstack((data['x'], data['y'], data['z'])).T
# uint8
sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
#float 32
sub_labels = data['scalar_Class']
print('test')
