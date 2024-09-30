import os
import rospy
import cv2
import imutils
import numpy as np
from cv_bridge import CvBridge
from imutils.object_detection import non_max_suppression


class IMGProcess:
    def __init__(self):
        self.bridge = CvBridge()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # def __undistort(self, frame, ci_msg):
    #     h, w = frame.shape[:2]
    #     K, D = np.array(ci_msg.K).reshape(3, 3), np.array(ci_msg.D)
    #     new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (h, w), 0, (h, w))
    #     mapX, mapY = cv2.initUndistortRectifyMap(K, D, None, new_camera_mtx, (h, w), 5)
    #     img = cv2.remap(frame, mapX, mapY, cv2.INTER_LINEAR)
    #     return img
    
    def __get_bbox_area(self, bbox):
        x1, x2 = bbox[0], bbox[2]
        y1, y2 = bbox[1], bbox[3]
        return (x2 - x1) * (y2 - y1)

    def __get_biggest_bbox(self, bboxes):
        areas = [self.__get_bbox_area(bbox) for bbox in bboxes]
        return bboxes[np.argmax(areas)]

    def preprocess(self, img_msg):
        """
        Process image before running detection algorithm.

        Parameters:
        - img_msg: Image message from camera containing current frame

        Returns:
        - Processed image
        """

        print("Processing Image...")
        if img_msg is None:
            return None
        
        frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        # Rotate depending on whether the turtlebot has inverted camera or not...
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Try undistorting the image from curved image due to fisheye lens to normal flat image
        # frame = self.__undistort(frame, ci_msg)

        return frame

    def detect(self, image):
        """
        Runs algorithm for detecting person in current frame

        Parameters:
        - image: Current frame potentially containing people

        Returns:
        - The biggest bounding box in the frame (biggest because algorithm sometimes 
            detects smaller objects thinking they are people)
        """

        image = imutils.resize(image, width=min(400, image.shape[1]))
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4),
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

        return self.__get_biggest_bbox(pick) if pick != [] else None
