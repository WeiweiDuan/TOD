# training data generation

import cv2
import numpy as np
from utils import load_data, data_augmentation, helper
import os


MAP_PATH = './data/orcutt_2018/CA_Orcutt_20180828_TM_geo.png'
MASK_PATH = './data/orcutt_2018/wetlands.png'
SUBSET_PATH = './data/orcutt_2018/neg_samples'

IMG_SIZE = 40
STRIDE = 20
num_imgs = 1000

helper.create_folder(SUBSET_PATH)

mask = cv2.imread(MASK_PATH)
print(mask.shape)
all_data, _ = load_data.load_all_data(MAP_PATH, MASK_PATH, IMG_SIZE, STRIDE, flip=True)
print(all_data.shape)
np.random.shuffle(all_data)
# helper.remove_files(SUBSET_PATH)
for i in range(num_imgs):
    cv2.imwrite(os.path.join(SUBSET_PATH,str(i)+'.png'), all_data[i].reshape((IMG_SIZE, IMG_SIZE, 3)))
print('done...')
