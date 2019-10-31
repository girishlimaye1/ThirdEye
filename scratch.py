from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

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


#
# '''
# Describe an image - remote
# This example describes the contents of an image with the confidence score.
# '''
# remote_image_url='https://homepages.cae.wisc.edu/~ece533/images/airplane.png'
# print("===== Describe an image - remote =====")
# # Call API
# with utils.no_ssl_verification():
#     description_results = computervision_client.describe_image(remote_image_url, )
#
# # Get the captions (descriptions) from the response, with confidence level
# print("Description of remote image: ")
# if (len(description_results.captions) == 0):
#     print("No description detected.")
# else:
#     for caption in description_results.captions:
#         print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
#
#
#
'''
Recognize printed text - remote
This example will extract printed text in an image, then print results, line by line.
This API call can also recognize handwriting (not shown).
'''
print("===== Detect printed text - remote =====")
# Get an image with printed text

remote_image_printed_text_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/printed_text.jpg"

# Call API with URL and raw response (allows you to get the operation location)
with utils.no_ssl_verification():
    recognize_printed_results = computervision_client.batch_read_file(remote_image_printed_text_url,  raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = operation_location_remote.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                print(line.text)
                #print(line.bounding_box)
    print()
