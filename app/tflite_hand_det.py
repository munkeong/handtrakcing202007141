import tensorflow as tf
import numpy as np
import cv2
import json
from tflite_model import *

def _get_triangle(self, kp0, kp2, dist=1):
    """get a triangle used to calculate Affine transformation matrix"""

    dir_v = kp2 - kp0
    dir_v /= np.linalg.norm(dir_v)
    R90 = np.r_[[[0,1],[-1,0]]]
    dir_v_r = dir_v @ R90.T
    return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])

#read anchors
# import csv
# anchors_path = "anchors.csv"
# with open(anchors_path, "r") as csv_f:
#     anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]

POINT_COLOR = (10, 0, 225)
CONNECTION_COLOR = (0, 0, 0)
THICKNESS = 2
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

#init tflite model
# model_palm  = Model("palm_detection.tflite")
# in_shape = model_palm.getInputShape()
# h = in_shape[1]
# w = in_shape[2]

# #read input image
# in_img = "hand.jpeg"
# img = cv2.imread(in_img)
# img_reszd = cv2.resize(img, (w, h))
# img_pre = ((img_reszd - 127.5) /  127.5).astype('float32')

# #run palm detection model
# outputs_palm = model_palm.runModel(img_pre)
# regressors = outputs_palm[0][0]
# classifiers = outputs_palm[1][0,:,0]
# print(regressors.shape)
# print(classifiers.shape)

#threshold outputs
# probs_tmp = sigm(classifiers)
# thresh = 0.3
# det_mask = probs_tmp > thresh
# det_cand = regressors[det_mask]
# anchor_cand = anchors[det_mask]
# probs = probs_tmp[det_mask]

# moved_candidate_detect = det_cand.copy()
# moved_candidate_detect[:, :2] = det_cand[:, :2] + (anchor_cand[:, :2] * 256)
# box_ids = non_max_suppression_fast(moved_candidate_detect[:, :4], probs)
# box_ids = box_ids[0]

# # bounding box offsets, width and height
# dx,dy,w,h = det_cand[box_ids, :4]
# center_wo_offst = anchor_cand[box_ids,:2] * 256
# box_enlarge = 0.5
# box_shift = 0.2

# 7 initial keypoints
# keypoints = center_wo_offst + det_cand[box_ids,4:].reshape(-1,2)
# side = max(w,h) * box_enlarge
# now we need to move and rotate the detected hand for it to occupy a
# 256x256 square
# line from wrist keypoint to middle finger keypoint
# should point straight up
# TODO: replace triangle with the bbox directly
# source = _get_triangle(keypoints[0], keypoints[2], side)
# source -= (keypoints[0] - keypoints[2]) * box_shift

# #draw output on image
# for point in keypoints:
#     x, y = point
#     cv2.circle(img_reszd, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
#     for connection in connections:
#         x0, y0 = keypoints[connection[0],:]
#         x1, y1 = keypoints[connection[1],:]
#         cv2.line(img_reszd, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

#init hand landmark det
model_hand_lm = Model("hand_landmark.tflite")
model_hand_lm_in = model_hand_lm.getInputShape()
h_hand = model_hand_lm_in[1]
w_hand = model_hand_lm_in[2]

# in_img = "hand.jpeg"
# img = cv2.imread(in_img)
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        img_reszd = cv2.resize(img, (w_hand, h_hand))
        img_pre_hand = ((img_reszd - 127.5) /  127.5).astype('float32')

        outputs_hand = model_hand_lm.runModel(img_pre_hand)

        # import time
        # for i in range(10):
        #     start = time.time()
        #     outputs_hand = model_hand_lm.runModel(img_pre_hand)
        #     end = time.time()

        #     print("elapsed time: %s" %(end-start))
            
        hand_joints = outputs_hand[0][0].reshape(-1,2) #21,2
        hand_flag = outputs_hand[1]


        #find ratio of input image and output tensor
        in_h = img.shape[0]
        in_w = img.shape[1]
        ratio_h = h_hand / in_h
        ratio_w = w_hand / in_w

        #append output x,y into list
        xs = []
        ys = []

        cv2.imshow("input",img)
        if hand_flag:
            for i in range(hand_joints.shape[0]):
                x = hand_joints[i,0] / ratio_w
                y = hand_joints[i,1] / ratio_h
                xs.append(x)    
                ys.append(y)

                #draw joint location on image
                cv2.circle(img, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
                for connection in connections:
                    x0, y0 = hand_joints[connection[0],:]
                    x1, y1 = hand_joints[connection[1],:]
                    x0 = x0 / ratio_w
                    x1 = x1 / ratio_w
                    y0 = y0 / ratio_h
                    y1 = y1 / ratio_h
                    #draw joint connections by line
                    cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, 1)

            #write output into json
            model_output_json = {}
            model_output_json['x'] = list(xs)
            model_output_json['y'] = list(ys)

            with open("hand_joints.json","w") as json_file:
                json.dump(model_output_json, json_file)
            
        cv2.imshow("output",img)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            img.release()
            break
    else:
        break