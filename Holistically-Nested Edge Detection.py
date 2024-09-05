#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import cv2
import os
import numpy as np

# Define CropLayer class
class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        H, W = targetShape[2], targetShape[3]
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]

# Use argparse to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True,
                help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--image-folder", type=str, required=True,
                help="path to folder containing input images")
args = vars(ap.parse_args())

# Register CropLayer
cv2.dnn_registerLayer('Crop', CropLayer)

# Load HED model
protoPath = os.path.join(args["edge_detector"], "deploy.prototxt")
modelPath = os.path.join(args["edge_detector"], "hed_pretrained_bsds.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Get the path to the image folder
image_folder = args["image_folder"]

# Iterate over all files in the image folder
for image_name in os.listdir(image_folder):
    # Construct the full image path
    image_path = os.path.join(image_folder, image_name)
    
    # Check if the file is an image
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    # Read and preprocess the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        continue

    (h, w) = image.shape[:2]

    # Resize the image to fit the HED model
    scale = 500.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Convert the image to a blob
    blob = cv2.dnn.blobFromImage(resized_image, scalefactor=1.0, size=(new_w, new_h),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)

    # Set the blob as the input to the network and perform a forward pass
    net.setInput(blob)
    hed = net.forward()

    # Resize the result and save it
    hed = cv2.resize(hed[0, 0], (new_w, new_h))
    hed = (255 * hed).astype("uint8")
    final_hed = cv2.resize(hed, (w, h))  # Scale the result back to the original size

    # Construct the save path and save the result
    output_path = os.path.join(image_folder, f"hed_{image_name}")
    cv2.imwrite(output_path, final_hed)

    # Display the original image and the edge detection result
    cv2.imshow("Input", resized_image)
    cv2.imshow("HED", final_hed)
    cv2.waitKey(1)  # Add a short delay to display each image

# Close all OpenCV windows
cv2.destroyAllWindows()


# In[ ]:




