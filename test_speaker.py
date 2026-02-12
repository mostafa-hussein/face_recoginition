import socket
import time

SERVER_IP = "192.168.50.40"
PORT = 65432
MESSAGE = "Hi suzan. Akash is trying to wake up. Please check on him."
MESSAGE = "Hi Dad. It seems like you need some help with your coffee. I will be there in a moment."

# MESSAGE = "true"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
    s.connect((SERVER_IP, PORT))
    s.sendall(MESSAGE.encode('utf-8'))

time.sleep(10)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
    s.connect((SERVER_IP, PORT))
    s.sendall(MESSAGE.encode('utf-8'))

time.sleep(10)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
    s.connect((SERVER_IP, PORT))
    s.sendall(MESSAGE.encode('utf-8'))
