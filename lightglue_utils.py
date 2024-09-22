import os
import cv2 
import math
import torch
import kornia as K
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
#matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).eval().to(device)
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
#matcher.compile(mode='reduce-overhead')

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (640, 480))  # 필요한 경우 이미지 크기 조정
    image = K.image_to_tensor(image, False).float() / 255.0
    image = image.to(device)
    return image

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
        # img1 = stabilizer.stabilize_frame(img1)
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
    
def get_errors(coord_list, float_origin_coordinate, len_coord, len_videos):
    misannotate_error = 0
    pixel_error = 0
    
    for index in range(len_videos):
        for i in range(len_coord):
            try:
                origin_x = float_origin_coordinate[index][i][0]
                origin_y = float_origin_coordinate[index][i][1]
                
                _coord = coord_list[index][i]
                
                x = _coord[0][0]
                y = _coord[0][1]
                
                x = x / 640
                y = y / 480
                
                x = round(x, 4)
                y = round(y, 4)
                
                distance = math.sqrt((origin_x - x)**2 + (origin_y - y)**2)
                
                if distance > 0.1:
                    misannotate_error += 1
                
                if distance > pixel_error:
                    pixel_error = distance
            except:
                pass
                
    return misannotate_error, pixel_error