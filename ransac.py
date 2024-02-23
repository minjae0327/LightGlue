import random
import math
import numpy as np

def ransac(target_keypoint, frame_keypoint):
    best_homography = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    if len(target_keypoint) < 6:
        return best_homography, 0
    
    max_iteration = 1000
    iteration = 1000
    iteration_count = 0
    p = 0.99
    s = 4
    best_inlier_rate = -1
    
    while 1:
        if iteration_count >= iteration:
            break
        elif iteration_count >= max_iteration:
            break
        
        target_sample = []
        frame_sample = []
        
        #호모그래피 추정에 필요한 최소 4개의 특징점을 뽑음
        for _ in range(s):
            idx = random.randint(0, len(target_keypoint)-1)
            target_sample.append(target_keypoint[idx])
            frame_sample.append(frame_keypoint[idx])
            
        homography = find_homography(target_sample, frame_sample)
        inliers = calculate_inliers(homography, target_keypoint, frame_keypoint)
    




def find_homography(src_point, dst_point):
    A = []
    for i in range(4):
        x, y = src_point[i]
        u, v = dst_point[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)  #U, Sigma, Vh
    L = Vh[-1:] / Vh[-1, -1]
    H = L.reshape(3, 3)
    
    return H


def calculate_inliers(homography, target_keypoint, frame_keypoint):
    