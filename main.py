import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import cv2
import imutils
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

from img_process import IMGProcess
from follower import FollowObject


class ObjectTracker:
    def __init__(self):
        print("Initializing...\n")
        rospy.init_node('object_tracker', anonymous=True)
        img_sub = Subscriber('/camera/image', Image)
        scan_sub = Subscriber('/scan', LaserScan)
        # ApproximateTimeSynchronizer makes sure the two messages are received at 
        # approximately the same time (syncs the transmission time of the two topics)
        ts = ApproximateTimeSynchronizer([img_sub, scan_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.callback)

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.frame_width = 0
        self.img_process = IMGProcess()
        self.follower = FollowObject()
        self.try_follow = False

    def callback(self, img_msg, scan_msg):
        """
        Intermediary callback function to start the program. 
        ROS sends message from the camera and the LDS to this function, which calls the respective callbacks
        for the image and the distance messages separately.

        Parameters:
        - img_msg: Image message from camera containing current frame
        - scan_msg: LaserScan message from LDS containing distance values
        """

        self.image_callback(img_msg)
        if self.try_follow:
            self.follow_callback(scan_msg)
    
    def image_callback(self, img_msg):
        """
        Runs the object detection algorithm on current frame after preprocessing

        Parameters:
        - img_msg: Image message from camera containing current frame
        """

        try:
            img = self.img_process.preprocess(img_msg)
            self.frame_width = img.shape[1]
            object_bbox = self.img_process.detect(img)
            # Tracks and follows if there are any objects detected in the current frame
            if object_bbox is not None:
                self.try_follow = True
                self.track_object(object_bbox)
        except Exception as e:
            rospy.logerr(e)
    
    def follow_callback(self, scan_msg):
        """
        Runs the falgorithm to follow detected person/object

        Parameters:
        - scan_msg: LaserScan message from LDS containing distance values
        """

        print("Following...")
        # Make sure try_follow is True, implying there is an object detected and robot should follow it
        assert self.try_follow
        vel_msg = self.follower.follow(scan_msg)
        self.cmd_vel.publish(vel_msg)

    def track_object(self, bbox):
        """
        Method for tracking detected object in current frame. Reorients robot to center the detected object

        Parameters:
        - bbox: List containing the coordinates of bounding box of detected person/object
        """

        print("Orienting...")
        object_center_x = (bbox[0] + bbox[2]) / 2  # Calculate center of the object
        difference = object_center_x - self.frame_width / 2  # Calculate difference from frame center
        twist = Twist()

        if difference > 15:  # Threshold for right turn
            twist.angular.z = -0.1  # Angular velocity for turning right
        elif difference < -15:  # Threshold for left turn
            twist.angular.z = 0.1  # Angular velocity for turning left
        else:
            twist.angular.z = 0.0 
        self.cmd_vel.publish(twist)


if __name__ == '__main__':
    try:
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
