import cv2
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import (
    TextOperationStatusCodes,
)
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



colorMap={'NoHat':(0,0,255),'WithHat':(0,255,0),'NoCover':(0,0,255),'WithCover':(0,255,0)}
import utils
endpoint="https://westus2.api.cognitive.microsoft.com/"
pred_key ='fae1b9653c9942f5946d916910326f6a'
iteration='Iteration3'
proj_id='2a3536a6-894f-4f4e-85cb-6a82d84e0e54'



import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

predictor = CustomVisionPredictionClient(pred_key, endpoint=endpoint)
def callPrediction():

    with utils.no_ssl_verification():
        # Open the sample image and get ba3ck the prediction results.
        # frames = open("stream.jpg", encoding="utf8")
        # results = predictor.detect_image(proj_id, iteration,
        #                                      (frames.read()))
        with open("stream.jpg",mode="rb") as test_data:
            results = predictor.detect_image(proj_id, iteration,
                                             test_data)
    result_dict={}
    # Display the results.
    for prediction in results.predictions:
        # print(
        #     "\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, "
        #                                  "bbox.top = {2:.2f}, bbox.width = {"
        #                                  "3:.2f}, bbox.height = {4:.2f}".format(
        #         prediction.probability * 100, prediction.bounding_box.left,
        #         prediction.bounding_box.top, prediction.bounding_box.width,
        #         prediction.bounding_box.height))
        if prediction.probability >0.2:
            result_dict[prediction.tag_name]=[prediction.bounding_box.left,prediction.bounding_box.top, prediction.bounding_box.width,
                    prediction.bounding_box.height,prediction.probability ]
            print(result_dict)


    return result_dict
def gen():

    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FPS, 0.1)
    while True:
        ret, image_np = cap.read()

        #Write image to file
        cv2.imwrite("stream.jpg", image_np)

        #

        result_dict = callPrediction()


        # if object found
        for objecttype, values in result_dict.items():
            print(image_np.shape)
            img_width=image_np.shape[0]
            img_height=image_np.shape[1]
            top_left=(int(values[0]*img_height),int(values[1]*img_width) )
            bottom_right=(int(img_height*(values[0]+values[2])),int(img_width*(values[1]+values[3])))
            #bottom_right=(630,470)
            color=(0, 0, 255)
            print(top_left,bottom_right,color)

            rectangle = cv2.rectangle(image_np, top_left, bottom_right, colorMap[objecttype], 5)
            #rectangle = cv2.rectangle(image_np, top_left , bottom_right,color , 5)
            #Call model
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = cv2.putText(
                image_np, objecttype, top_left, font, 1, colorMap[objecttype], 2, cv2.LINE_AA
            )

        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        #frames = open("stream.jpg", "wb+")

        #Get this uncommented  if you want flask


        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    gen()
    #app.run(host="0.0.0.0", debug=False, threaded=True, port=3001)
