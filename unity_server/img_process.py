import os
import sys
import cv2 
import copy
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lightglue_utils import *
from CSRansac import csransac


def process_image(img, cracked_image=r"C:\Users\ailab\LightGlue\unity_server\CrackedImage.PNG"):
    # 이미지 파일이 존재하는지 확인
    # if not os.path.exists(image_path):
    #     print(f"Image file does not exist: {image_path}")
    #     return

    # OpenCV를 사용하여 이미지 읽기
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함하여 이미지 읽기
    # if img is None:
    #     print(f"Failed to load image at {image_path}")
    #     return
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 전처리 하기
    results_lightglue = matching_keypoints(cracked_image, image)
    cracked_keypoint = results_lightglue["points0"].cpu().numpy()
    video_keypoint = results_lightglue["points1"].cpu().numpy()
    
    H, inliers = csransac(cracked_keypoint, video_keypoint)
    if np.mean(inliers) >= 0.8:
        return "Crack Detected"
    
    return "No Crack Detected"


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python image_processor.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     cracked_image = r"C:\Users\ailab\LightGlue\unity_server\CrackedImage.PNG"
#     process_image(image_path, cracked_image)