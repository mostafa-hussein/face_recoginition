#!/bin/bash
xhost +SI:localuser:root
source /opt/ros/humble/setup.bash
source /home/carl/.bashrc
ros2 run rmw_zenoh_cpp rmw_zenohd &
export ROS_DOMAIN_ID=25
echo "DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY  USER=$USER"
/usr/bin/python3 /home/carl/projects/face_recoginition/brenda_tracker.py