
import cv2
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
import os
import sys
import time

from flask import Flask, render_template, Response
import cv2

port = 3030

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



import utils
print(os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY'])
print(os.environ['COMPUTER_VISION_ENDPOINT'])
# Add your Computer Vision subscription key to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = "bf00d070b31542af8ee0e6d62a2f0e47" #os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint ="https://westus.api.cognitive.microsoft.com/" #os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()


computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

import cv2
def gen():
    cap = cv2.VideoCapture(1)
    while True:
        ret, image_np = cap.read()



        #if object found
        rectangle = cv2.rectangle(image_np, (384, 0), (510, 128), (0, 0, 255), 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = cv2.putText(image_np, 'man', (10, 500), font, 4, (255, 255, 255), 2,
                           cv2.LINE_AA)
        #cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        frames = open("stream.jpg", 'wb+')

        if ret:  # frame captures without errors...
            cv2.imwrite("stream.jpg", image_np)  # Save image...



        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frames.read()) + b'\r\n')


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()

            cv2.destroyAllWindows()

            break

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True)
