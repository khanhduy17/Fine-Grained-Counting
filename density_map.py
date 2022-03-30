import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import json

gt_point_file = "/mmlabworkspace/WorkSpaces/KhanhND/Fine-Grained-Counting/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/annotations.json"
with open(gt_point_file, 'r') as f:
  points = json.load(f)

root = "/mmlabworkspace/WorkSpaces/KhanhND/Fine-Grained-Counting/Fine-Grained-Counting-Dataset/"  

for img_name, data in points.items():
    img_path = os.path.join(root, 'Towards_vs_Away/images', img_name)
    img = plt.imread(img_path)
    print(img_path)
    gt1_y = data[0]['y']
    gt1_x = data[0]['x']
    gt2_y = data[1]['y']
    gt2_x = data[1]['x']
  
    k1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    count1 = 0
    for i in range(len(gt1_y)):
        if int(gt1_y[i]) < img.shape[0] and int(gt1_y[i]) >= 0 and int(gt1_x[i]) < img.shape[1] and int(gt1_x[i]) >= 0:
            k1[int(gt1_y[i]), int(gt1_x[i])] = 1
            count1 += 1
    print('Ignore {} wrong annotation.'.format(len(gt1_y) - count1))
    k1 = gaussian_filter(k1, 5)
    
    k2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    count2= 0
    for i in range(len(gt2_y)):
        if int(gt2_y[i]) < img.shape[0] and int(gt2_y[i]) >= 0 and int(gt2_x[i]) < img.shape[1] and int(gt2_x[i]) >= 0:
            k2[int(gt2_y[i]), int(gt2_x[i])] = 1
            count2 += 1
    print('Ignore {} wrong annotation.'.format(len(gt2_y) - count2))
    k2 = gaussian_filter(k2, 5)
    
    k = np.stack((k1, k2), axis=0)
    
    print(k.shape)
    
    with h5py.File(img_path.replace('.png', '.h5').replace('images', 'ground-truths'), 'w') as hf:
        hf['density'] = k