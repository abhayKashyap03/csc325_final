# CSC325 Final Project

## Person Tracker and Follower

- Detects human in image
- Follows human when they move away from the robot past a certain distance


## Dependencies

- opencv
- rospy
- numpy
- imutils
- cv_bridge


## Usage

- Start `roscore` on desktop PC
- `ssh ubuntu@turtlebot3_xx.union.edu` to log in to turtlebot
- Start sensors and bringup on turtlebot by running:
<p style="text-align: center;"><code>roslaunch turtlebot3_bringup turtlebot3_robot.launch</code></p>

- Start camera on turtlebot by running:
<p style="text-align: center;"> <code>roslaunch turtlebot3_autorace_camera raspberry_pi_camera_publish.launch</code></p>

- Start detection, tracking and follower by running the following code on the desktop terminal:
<p style="text-align: center;"><code>python main.py</code></p>


## File Structure

#### Videos
- <b>detect_track.mp4:</b> Video of detection of person and tracking the detection in image
- <b>following.mp4:</b> Video of robot following the detected person in frame (moving towards detected person)

#### Code Files
- <b>main.py:</b> Main file that initializes and calls the other algorithms
- <b>following.py:</b> Code implementing the algorithm for following the detected person/object
- <b>img_process.py:</b> Code implementing object detection (person detection in this case) and other image preprocessing methods
- <b>initial_test_detect.py:</b> File containing some inital tests to see which detection algorithm works better
