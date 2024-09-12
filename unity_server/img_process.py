import os
import sys
import cv2 
import math
import copy
import torch
import numpy as np
import kornia as K
import kornia.feature as KF

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd, load_image_from_path
import CSRansac



def preprocess_image(image_data):
    npimg = np.frombuffer(image_data, np.uint8)  # 바이트 데이터를 numpy 배열로 변환
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # numpy 배열을 이미지로 디코딩
    
    return image


def show_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(5)
    cv2.destroyAllWindows()