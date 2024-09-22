# import os
# import sys
# import cv2 
# import math
# import copy
# import torch
# import numpy as np
# import kornia as K
# import kornia.feature as KF

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from lightglue import LightGlue, SuperPoint
# from lightglue.utils import load_image, rbd, load_image_from_path
# import CSRansac



# def preprocess_image(image_data):
#     npimg = np.frombuffer(image_data, np.uint8)  # 바이트 데이터를 numpy 배열로 변환
#     image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # numpy 배열을 이미지로 디코딩
    
#     return image


    
    
# print("Hello")

import socket
import numpy as np
import cv2

def preprocess_image(image_data):
    npimg = np.frombuffer(image_data, np.uint8)  # 바이트 데이터를 numpy 배열로 변환
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # numpy 배열을 이미지로 디코딩
    return image

# 소켓 서버 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 5001))  # IP와 포트를 바인딩
server_socket.listen(1)
print("Waiting for Unity connection...")

# Unity 클라이언트와 연결
client_socket, addr = server_socket.accept()
print(f"Connected by {addr}")

try:
    while True:
        # 먼저 이미지 크기(4바이트)를 받음
        data = client_socket.recv(4)
        if not data:
            break
        image_size = int.from_bytes(data, byteorder='little')

        # 이미지 데이터를 받음
        image_data = b""
        while len(image_data) < image_size:
            packet = client_socket.recv(image_size - len(image_data))
            if not packet:
                break
            image_data += packet

        # 수신한 데이터를 이미지로 디코딩
        image = preprocess_image(image_data)

        # 수신한 이미지를 화면에 출력 (OpenCV 사용)
        cv2.imshow('Received Frame', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
