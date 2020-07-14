import os
from PIL import Image
from flask import Flask, request, Response, render_template
import cv2
import io
from tflite_model import *
from camera import VideoCamera
from camera import model_hand_lm
import json
app = Flask(__name__)

#init tflite model
model_hand = Model("hand_landmark.tflite")
in_shape = model_hand.getInputShape()
h_hand = in_shape[1]
w_hand = in_shape[2]

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response



@app.route('/')
def index():
    # return Response('ETRI Object Detection Test 2019.09.27 #8')
    return render_template('index.html')


def gen(cam_):
    while True:
        #get camera frame
        send_im = 0#0:output image, 1: json
        frame_output = cam_.getFrame(send_im)
        if send_im:
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_output + b'\r\n\r\n')
        else:
            #example
            print (frame_output)
            print("aegaweg")



@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


@app.route('/test')
def test():
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

    img = cv2.imread(TEST_IMAGE_PATHS)
    img_reszd = cv2.resize(img, (w_hand, h_hand))
    img_pre_hand = ((img_reszd - 127.5) /  127.5).astype('float32')
    output_tensors = model_hand_lm.runModel(img_pre_hand)
    output_json = hand_json(output_tensors, [img.shape[0],img.shape[1]], [h_hand, w_hand])
    return output_json


@app.route('/image', methods=['POST'])
def image():
    try:

        image_file = request.files['image'].read()  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        #if threshold is None:
        #  threshold = 0.5
        #else:
        #  threshold = float(threshold)

        img = np.array(Image.open(io.BytesIO(image_file)))
        #img = cv2.imread(image_file)
        img_reszd = cv2.resize(img, (w_hand, h_hand))
        img_pre = ((img_reszd - 127.5) /  127.5).astype('float32')
        print(img_pre.shape) 
        output_tensors = model_hand.runModel(img_pre[:,:,0:3])
        output_json = hand_json(output_tensors, [img.shape[0],img.shape[1]], [h_hand, w_hand])
        return output_json

    except Exception as e:
        print('POST /image error: %e' % e)
        return e



if __name__ == '__main__':
	# without SSL
     app.run(debug=True, host='0.0.0.0')
    # app.run(debug=True)
	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
