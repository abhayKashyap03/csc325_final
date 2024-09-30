#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

# Load the full body cascade classifier
fullbody_cascade = cv2.CascadeClassifier('object_detect/haarcascade_fullbody.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_person(img):
    print("Detecting...")
    img = cv2.rotate(img, cv2.ROTATE_180)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_detect = fullbody_cascade.detectMultiScale(gray_img)

    for (x, y, w, h) in img_detect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Adjust waitKey value as needed

def detect2(img):
    print("Detecting...")
    image = cv2.rotate(img, cv2.ROTATE_180)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    cv2.imshow("Image", image)
    cv2.waitKey(1)

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        print("Running detector... \n\n")
        detect2(cv_image)
    except Exception as e:
        rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('object_detection', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("/camera/image", Image, image_callback)
    rospy.spin()