import cv2
from tflite_model import *

model_hand_lm = Model("hand_landmark.tflite")
model_hand_lm_in = model_hand_lm.getInputShape()
h_hand = model_hand_lm_in[1]
w_hand = model_hand_lm_in[2]

class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def getFrame(self, img_flag):
        #extracting frames
        ret, frame = self.video.read()
        if img_flag:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            if ret:
                img_reszd = cv2.resize(frame, (w_hand, h_hand))
                img_pre = ((img_reszd - 127.5) /  127.5).astype('float32')
                output_tensors = model_hand_lm.runModel(img_pre)
                output_json = hand_json(output_tensors, [frame.shape[0],frame.shape[1]], [h_hand, w_hand])
                return output_json
            else:
                print("video read error")
                return None



