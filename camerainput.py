
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

import utils

# Add your Computer Vision subscription key to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()


computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, image_np = cap.read()


    #if object found
    rectangle = cv2.rectangle(image_np, (384, 0), (510, 128), (0, 0, 255), 5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = cv2.putText(image_np, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2,
                       cv2.LINE_AA)
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()

        cv2.destroyAllWindows()

        break