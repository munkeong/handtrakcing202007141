import tensorflow as tf
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from tflite_model import *
import json
#init model
model_hair = Model("hair_segmentation.tflite")
in_shape = model_hair.getInputShape()
h = in_shape[1]
w = in_shape[2]

#read image & preprocess
in_img = "hair.jpeg"
img = cv2.imread(in_img)
img_reszd = cv2.resize(img, (w, h))
img_pre = (img_reszd / 255 - 0.5) * 2

#1-3 channels: RGB
#4 channels: previous frame mask
in_tensor = np.zeros(in_shape[1:],dtype=np.float32)
in_tensor[:,:,0:3] = img_pre

outputs = model_hair.runModel(in_tensor)
output = np.squeeze(outputs)

# import time 
# for i in range(10):
#     start = time.time()
#     outputs = model_hair.runModel(in_tensor)
#     end = time.time()

#     print("elapsed time: %s" %(end-start))


for i in range(2):
    plt.imshow(output[:,:,i])
    plt.colorbar()
    plt.show()

thresh = 2.5
mask = np.where(output[:,:,1] > thresh, 1, 0)
mask_idx = np.nonzero(mask)
plt.imshow(mask)
plt.colorbar()
plt.show()

#write output into json
#append output x,y into list
xs = []
ys = []
for i in range(len(mask_idx[0])):
    xs.append(str(mask_idx[0][i]))
    ys.append(str(mask_idx[1][i]))
model_output_json = {}
model_output_json['x'] = xs
model_output_json['y'] = ys
with open("hair_segmented.json","w") as json_file:
    json.dump(model_output_json, json_file)

