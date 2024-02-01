#!/usr/bin/env python
# coding: utf-8

# # YOLOv8 Object Detection on Self-Driving-Car Data

# ![How-Tesla-Is-Using-Artificial-Intelligence-to-Create-The-Autonomous-Cars-Of-The-Future.jpg](attachment:How-Tesla-Is-Using-Artificial-Intelligence-to-Create-The-Autonomous-Cars-Of-The-Future.jpg)

# ## What is YOLO?
# You Only Look Once (YOLO) is a cutting-edge object detection method that significantly speeds up and simplifies the process of identifying objects in images and videos. Unlike traditional approaches, YOLO treats object detection as a single step, predicting object positions and categories directly. By doing so, it achieves real-time detection without sacrificing accuracy. YOLO's neural network architecture processes images swiftly, making it valuable for applications like self-driving cars, surveillance, and robotics. YOLO's unique approach has revolutionized object detection by making it faster and more accessible while maintaining high performance.

# # Importing libraries:

# In[1]:


import numpy as np
import PIL 
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from glob import glob
import random
import cv2
import warnings
warnings.simplefilter('ignore')


# In[2]:


get_ipython().system('pip install ultralytics')


# YOLOv8 is a group of neural network models. These models were created and trained using PyTorch and exported to files with the .pt extension. In this project we use the yolov8m.pt which is a middle-sized model for object detection. All YOLOv8 models for object detection ship already pre-trained on the COCO dataset, which is a huge collection of images of 80 different types.

# In[3]:


import ultralytics
from ultralytics import YOLO
yolo_model = YOLO('yolov8m.pt')


# In this section we have loaded the self-driving-cars image dataset which is used for training and testing autonomous vehicle systems and is crucial for developing and evaluating the performance of self-driving algorithms and models. Then we have randomly selected some images to implement yolov8 model on them as samples.

# In[11]:


import cv2
import glob
import random
import matplotlib.pyplot as plt

root_path = 'C:/Users/master/OneDrive/Desktop/self driving car dataset/YOLOv8 Object Detection on Self-Driving-Car Data/images/*.jpg'
num_samples = 4
images_data = glob.glob(root_path)
random_image = random.sample(images_data, num_samples)

plt.figure(figsize=(10, 6))
for i in range(num_samples):
    plt.subplot(2, 2, i+1)
    img = cv2.imread(random_image[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with Matplotlib
    plt.imshow(img)
    plt.axis('off')

plt.show()


# In this section, we have extracted significant results from the pre-trained YOLOv8 model, including the 'name of the detected object,' 'bounding box coordinates', and 'detection probabilities'. The results for the samples have been calculated separately.

# In[8]:


# Create a list to store the images
images = []
for i in range(num_samples):
    yolo_outputs = yolo_model.predict(random_image[i])
    output = yolo_outputs[0]
    box = output.boxes
    names = output.names
    
    for j in range(len(box)):
        labels = names[box.cls[j].item()]
        coordinates = box.xyxy[j].tolist()
        confidence = np.round(box.conf[j].item(), 2)
        #print(f'In this image {len(box)} objects has been detected.')
        print(f'Object {j + 1} is: {labels}')
        print(f'Coordinates are: {coordinates}')
        print(f'Confidence is: {confidence}')
        print('-------')
        
    # Store the image in the 'images' list
    images.append(output.plot()[:, :, ::-1])


# In the last section, the results for the samples have been presented visually..

# In[12]:


# plotting the images after object detection
print('\n\n-------------------------------------- Images after object detection with YOLOV8 --------------------------------')    

plt.figure(figsize=(10,6))
for i, img in enumerate(images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.axis('off')    
plt.tight_layout()
plt.show()

