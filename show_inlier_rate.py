import os
import cv2 
import json
import math
import torch
import numpy as np
from vidstab import VidStab

from lightglue import viz2d
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, load_image_from_path

import CSRansac
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
<<<<<<< Updated upstream
from matplotlib.animation import FFMpegWriter
=======
>>>>>>> Stashed changes

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)

aircraft_datasets = "datasets"
lables = os.path.join(aircraft_datasets + "/label")

stabilizer = VidStab()

def match_lightglue(img0, img1):
    img0 = load_image(img0)
    img1 = load_image(img1)

    # extract local features
    feats0 = extractor.extract(img0.to(device))  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(img1.to(device))
    
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    
    # get results
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = kpts0[matches[..., 0]]  # coordinates in img0, shape (K,2)
    points1 = kpts1[matches[..., 1]]  # coordinates in img1, shape (K,2)
        
    return {
        "points0": points0,
        "points1": points1,
    }

def matching_keypoints(target_img, video_img, stabilizing=False):
    # 이미지를 불러옴
    img0 = load_image(target_img, grayscale=True)
    if stabilizing == True:
        img1 = cv2.imread(video_img)
        img1 = stabilizer.stabilize_frame(img1)
        img1 = load_image(img1, grayscale=True)
    else:
        img1 = load_image(video_img , grayscale=True)

    # extract local features
    feats0 = extractor.extract(img0.to(device))  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(img1.to(device))

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    # get results
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = kpts0[matches[..., 0]]  # coordinates in img0, shape (K,2)
    points1 = kpts1[matches[..., 1]]  # coordinates in img1, shape (K,2)

    return {
        "points0": points0,
        "points1": points1,
    }

def get_origin_coordinate(origin_coordinate):
    # 원점 좌표값 불러오기
    for label_file in os.listdir(lables):
        label_path = os.path.join(lables, label_file)
        with open(label_path, "r") as f:
            json_file = json.load(f)
            coord = json_file["targetAnnotation"]
            coord[0] = coord[0] * 640
            coord[1] = coord[1] * 480
            origin_coordinate.append(coord)
            
    return origin_coordinate


def get_float_origin_coordinate(float_origin_coordinate):
    lables = os.path.join(aircraft_datasets + "/label")
    # 원점 좌표값 불러오기
    for label in os.listdir(lables):
        label_path = os.path.join(lables, label)
        with open(label_path, "r") as f:
            json_file = json.load(f)
            coord = json_file["targetAnnotation"]
            float_origin_coordinate.append(coord)
            
    return float_origin_coordinate


def get_inlier_rate():
    image0 = load_image_from_path("img0.png", grayscale=True)
    cap = cv2.VideoCapture('demo_video_resized.mp4')

    inlier_rates = []

    # 각 프레임 처리
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        feats0 = extractor.extract(image0.to(device))
        image1 = load_image(frame, grayscale=True)
        feats1 = extractor.extract(image1.to(device))
        
        matches01 = matcher({"image0": feats0, "image1": feats1})
        
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        homography, mask = CSRansac.csransac(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy())
        inlier_rates.append(mask)
        
        image0 = image1
        
    return inlier_rates

        

def main():
<<<<<<< Updated upstream
    # inlier_rates.json 파일에서 데이터 로드
    with open("inlier_rate.json", "r") as f:
        inlier_rates = json.load(f)
    
    # 그래프 설정
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r-', animated=True)  # 선 대신 점으로 표현
    
    def init():
        ax.set_xlim(0, 10)  # 초기 x축 범위 설정
        ax.set_ylim(0, 1)  # inlier rate의 범위는 0에서 1 사이라고 가정
        return ln,

    def update(frame):
        if inlier_rates[frame] >= 0:
            xdata.append(frame)
            ydata.append(inlier_rates[frame])
            ln.set_data(xdata, ydata)

        # x축의 최대값을 현재 프레임에 맞춰 동적으로 업데이트
        if frame >= ax.get_xlim()[1]:
            ax.set_xlim(ax.get_xlim()[0], frame + 10)  # 여기서 10은 추가적인 여유 공간입니다.
            ax.figure.canvas.draw()  # 캔버스를 다시 그려 x축 업데이트 반영

        return ln,

    ani = FuncAnimation(fig, update, frames=range(len(inlier_rates)),
                        init_func=init, blit=True, interval=(1000/30), repeat=False)
=======
    with open("inlier_rate.json", "r") as f:
        inlier_rates = json.load(f)
    
    len_inlier = len(inlier_rates)
    
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r-', animated=True)

    for i in range(len_inlier):
        def init():
            ax.set_xlim(0, i)  # 프레임 수에 따라 조절
            ax.set_ylim(0, 1)  # inlier rate의 최댓값에 따라 y축 범위 조절
            return ln,

        def update(frame):
            xdata.append(frame)
            ydata.append(inlier_rates[frame])
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, frames=i,
                            init_func=init, blit=True, interval=(1000/30))
>>>>>>> Stashed changes

    plt.xlabel('Frame')
    plt.ylabel('Inlier Rate')
    plt.title('Inlier Rate Over Time')
<<<<<<< Updated upstream
    
    # GIF로 저장하기 전에 imagemagick이 설치되어 있어야 함
    ani.save('lightglue_inlier_rate.gif', writer='imagemagick', fps=30)
    plt.show()

if __name__ == "__main__":
    main()
=======
    plt.show()
    



if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
