import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import pandas as pd

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_density(img_path, gt):
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', gt).replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gt = mat["image_info"][0, 0][0, 0][0]
    count = 0
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][1]) >= 0 and int(gt[i][0]) < img.shape[1] and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    print('Ignore {} wrong annotation.'.format(len(gt) - count))
    k = gaussian_filter(k, 5)
    df = pd.DataFrame(k)
    inf_count = np.isinf(df).values.sum()
    nan_count = np.isnan(k).sum()
    print("It contains " + str(inf_count) + " infinite values" + str(nan_count) + " NaN values")
    return k, count, inf_count, nan_count

root = "/mmlabworkspace/WorkSpaces/KhanhND/Fine-Grained-Counting/Fine-Grained-Counting-Dataset/Mask_vs_NoMask"

NFM_train = os.path.join(root, 'train_data', 'images')

train_file_list = './train_list_rambalac_1k.txt'

with open(train_file_list) as f:
    img_paths = f.readlines()
img_paths = [l.strip('\n\r') for l in img_paths]

for img_path in img_paths:
    img_path = os.path.join(NFM_train, img_path)
    print(img_path)
        
    k_mask, count_mask, inf_count_mask, nan_count_mask = compute_density(img_path, 'ground_truth_mask')
    k_nomask, count_nomask, inf_count_nomask, nan_count_nomask = compute_density(img_path, 'ground_truth_nomask')
    
    if inf_count_mask > 0 or inf_count_nomask > 0:
        break
    if nan_count_mask > 0 or nan_count_nomask > 0:
        break
    
        
    k = np.stack((k_mask, k_nomask), axis=0)
    
    ensure_dir(img_path.replace('.jpg', '.h5').replace('images', 'ground-truths'))
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truths'), 'w') as hf:
        hf['density'] = k
        
