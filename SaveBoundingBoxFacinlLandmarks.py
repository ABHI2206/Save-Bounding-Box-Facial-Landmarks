
# coding: utf-8

# In[1]:


# import os
# import dlib
# import glob
# import cv2
# import imutils
# from imutils import face_utils
# import numpy as np
# import argparse
# import imageio

# predictor_path = "./shape_predictor_68_face_landmarks.dat"

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# # img =  imageio.imread("Aaron_Eckhart_0001.jpg")
# img =  imageio.imread('./AFLW/0030-image06011.jpg')

# # image = imutils.resize(img, width=500)
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# # detect faces in the grayscale image
# rects = detector(img, 1)

# # loop over the face detections
# for (i, rect) in enumerate(rects):
#     # determine the facial landmarks for the face region, then
#     # convert the landmark (x, y)-coordinates to a NumPy array
#     shape = predictor(img, rect)
#     shape = face_utils.shape_to_np(shape)
#     print(shape.shape)
# # loop over the face parts individually
#     for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
#         # clone the original image so we can draw on it, then
#         # display the name of the face part on the image
#         clone = img.copy()
#         cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
 
#         # loop over the subset of facial landmarks, drawing the
#         # specific face part
#         for (x, y) in shape[i:j]:
#             cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

#             # extract the ROI of the face region as a separate image
#         (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
#         print(x, y, w, h)
#         if(x < 0):
#             x = 0
#         if(y < 0):
#             y = 0
#         roi = img[y:y + h, x:x + w]
#         roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
 
#         # show the particular face part
#         cv2.imshow("ROI", roi)
#         cv2.imshow("Image", clone)
#         cv2.waitKey(0)
 
#     # visualize all facial landmarks with a transparent overlay
#     output = face_utils.visualize_facial_landmarks(img, shape)
#     cv2.imshow("Image1", output)
#     cv2.waitKey(0)


# In[ ]:


# Code to SAVE Images

import os
import dlib
import glob
import cv2
import imutils
from imutils import face_utils
import numpy as np
import argparse
import pandas as pd
from PIL import Image
from PIL import Image
from scipy import misc
import imageio
import numpy as np
import matplotlib.pyplot as plt


predictor_path = "./shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./AFLW/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

from skimage import io
all_files = os.listdir(faces_folder_path)
# print(faces_folder_path)

for f in all_files:
    print(f)
    img =  imageio.imread(faces_folder_path + f)
    
    rects = detector(img, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        print("i ",i)
     
        (x, y, w, h) = cv2.boundingRect(np.array(shape))
        print(x,y,w,h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0

        roi = img[y:y + h, x:x + w]
        roi = misc.imresize(roi , (224,224) )
        
        imageio.imwrite('ByMe_AFLW_BoundingBox/' + f, roi)