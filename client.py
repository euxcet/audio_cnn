import socket
import time
import os
import struct

def gen():
    while True:
        yield os.urandom(100)

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect(('127.0.0.1', 8001))
for x in gen():
    socket.send(x)
