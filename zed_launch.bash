#!/bin/bash
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"

sleep 1
ros2 launch zed_wrapper zed2i.launch.py &

sleep 3

# Create a timestamp for log file naming
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the log file path with timestamp
LOG_FILE="/home/jetson/face_recoginition/logs/face_recoginition_$TIMESTAMP.log"

# Start Python script with logging to both terminal and file
python3 /home/jetson/face_recoginition/face_recoginition_jetson.py  --db /home/jetson/face_recoginition/face_database_lab_2.pkl --save_image 2>&1 | tee "$LOG_FILE" &


# Wait for commands to finish
wait







