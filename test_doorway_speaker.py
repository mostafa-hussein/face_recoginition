import socket

SERVER_IP = '192.168.50.95' # Jetson’s address
PORT = 65433
# MESSAGE = "Hi Dad. Are you going out? ... Make sure you wear your sneakers."
MESSAGE = "true" # If sent "true" string, it will play pre-recorded audio

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, PORT))
    s.sendall(MESSAGE.encode('utf-8'))