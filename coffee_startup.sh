#!/bin/bash

source /opt/ros/humble/setup.bash &&
export ROS_DOMAIN_ID=25 &&

# Create a timestamp for log file naming
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") &&

# Define the log file path with timestamp
LOG_FILE="/home/jetson/projects/face_recoginition/logs/coffee_triger_$TIMESTAMP.log" &&

# Start Python script with logging to both terminal and file
python3 /home/jetson/projects/face_recoginition/face_rec.py  --db /home/jetson/projects/face_recoginition/face_database_lab_2.pkl --save_image 2>&1 | tee "$LOG_FILE" &

# Wait for commands to finish
wait






