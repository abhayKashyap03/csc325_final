from geometry_msgs.msg import Twist
import numpy as np


class FollowObject:
    def follow(self, scan_msg):
        """
        Runs alorithm for following detected person

        Parameter:
        - scan_msg: LaserScan message from LDS containing distance values

        Returns:
        - Twist message containing velocity commands
        """

        dir_range = list(range(0, 10)) + list(range(350, 360))
        dist_range = [scan_msg.ranges[idx] for idx in dir_range]
        print(dist_range, any(3.5 > dist > 2 for dist in dist_range), '\n')

        vel_msg = Twist()

        if any(3 > dist > 2 for dist in dist_range):
            vel_msg.linear.x = 0.3
        else:
            vel_msg.linear.x = 0
        
        return vel_msg
