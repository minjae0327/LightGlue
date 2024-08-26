import random
import math
import numpy as np
    
def csransac(target_keypoint, frame_keypoint):

    # input: np.array([[t_x1,t_y1],[t_x2,t_y2],...]), np.array([[f_x1,f_y1],[f_x2,f_y2],...])
    # [t_x1,t_y1] 과 [f_x1,f_y1]은 매칭된 특징점.
    # 매칭된 특징점의 인덱스가 같아야함.

    homography = [[1,0,0],[0,1,0],[0,0,1]]
    best_homography = [[1,0,0],[0,1,0],[0,0,1]]
    if len(target_keypoint) < 6: # 매칭된 특징점의 수가 5개 이하일 경우 호모그래피를 추정할 수 없다고 판단.
        return np.array([[1,0,0],[0,1,0],[0,0,1]]), 0
    max_iteration = 1000
    iteration = 1000
    iteration_count = 0
    max_g = 1000
    p = 0.99
    s = 4
    best_inlier_rate = -1
    while 1:
        if iteration_count >= iteration:
            break
        elif iteration_count >= max_iteration:
            break
        satisfaction = False
        count = 0
        while satisfaction == False:
            target_sample = list()
            frame_sample = list()
            grid_list = list()
            for i in range(34):
                grid_list.append(-1)
            for i in range(4):
                count += 1
                if count > max_g:
                    for top_idx in range(4):
                        target_sample.append(target_keypoint[top_idx])
                        frame_sample.append(frame_keypoint[top_idx])
                    break
                idx = random.randint(0, len(target_keypoint)-1)
                target_sample.append(target_keypoint[idx])
                frame_sample.append(frame_keypoint[idx])
                col, row = get_position(frame_keypoint[idx])
                grid_list[row] = col
            #==================================================
            #CSP
            m_i = 0
            m_j = 0
            for i in range(34):
                m_i = i
                if (grid_list[i] == -1):
                    continue
                for j in range(34):
                    m_j = j
                    if i == j:
                        continue
                    if (grid_list[j] == -1):
                        continue
                    if grid_list[i] == grid_list[j]:
                        break
                    d = i - j
                    if (grid_list[i] == (grid_list[j] - d)) or (grid_list[i] == (grid_list[j] + d)):
                        break
                    if d == 1 or d == -1:
                        if grid_list[i] == grid_list[j] + 2 or grid_list[i] == grid_list[j] - 2:
                            break
                    if d == 2 or d == -2:
                        if grid_list[i] == grid_list[j] + 1 or grid_list[i] == grid_list[j] - 1:
                            break
                if m_j != 33:
                    break
            if m_i == 33:
                satisfaction = True
            #==================================================
        
        homography = find_homography(target_sample, frame_sample)
        inliers = calculate_inliers(homography, target_keypoint, frame_keypoint,5)
        inlier_rate = inliers / len(target_keypoint)
        if inlier_rate == 1:
            break
        if inlier_rate > best_inlier_rate:
            best_inlier_rate = inlier_rate
            best_homography = homography
        e = 1 - best_inlier_rate
        try:
            iteration = math.log(1.0 - p) / math.log(1.0 - math.pow(1.0 - e, s))
        except:
            iteration = iteration
        iteration_count += 1
    if best_inlier_rate <= 0.3: # best_inlier_rate <= 0.3 일 경우 호모그래피 추정에 실패했다고 판단.
        return np.array([[1,0,0],[0,1,0],[0,0,1]]), best_inlier_rate
    return best_homography, best_inlier_rate
	
def get_position(keypoint):
    col_size = 34 # csp ransac grid size
    row_size = 34 # csp ransac grid size
    col = int(keypoint[0] / (640 / col_size)) # 원본 이미지 width
    row = int(keypoint[1] / (480 / row_size)) # 원본 이미지 height
    if col == col_size:
        col -= 1
    if row == row_size:
        row -= 1
    return (col, row)
	
def perspective_transform(src, homo):
    w = homo[2, 0] * src[0] + homo[2, 1] * src[1] + homo[2, 2]
    x = (homo[0, 0] * src[0] + homo[0, 1] * src[1] + homo[0, 2]) / w
    y = (homo[1, 0] * src[0] + homo[1, 1] * src[1] + homo[1, 2]) / w
    return (x, y)

def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

def find_homography(src_points, dst_points):
    A = []
    for i in range(4):
        X, Y = src_points[i][0], src_points[i][1]
        x, y = dst_points[i][0], dst_points[i][1]
        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])

    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def calculate_inliers(H, points1, points2, threshold=5):
    num_points = len(points1)
    points1_hom = np.concatenate([points1, np.ones((num_points, 1))], axis=1).T
    points2 = np.array(points2)
    estimates = np.dot(H, points1_hom)
    estimates /= estimates[2, :]
    errors = np.sqrt(np.sum((points2.T - estimates[:2, :]) ** 2, axis=0))
    inliers = np.sum(errors <= threshold)
    return inliers
    
def estimate_scale_change(old_points, new_points):
    old_center = np.mean(old_points, axis=0)
    new_center = np.mean(new_points, axis=0)
    old_dists = np.linalg.norm(old_points - old_center, axis=1)
    new_dists = np.linalg.norm(new_points - new_center, axis=1)
    return np.mean(new_dists) / np.mean(old_dists)


#------------ransac----------------
def ransac(target_keypoint, frame_keypoint):
    homography = np.array([[1,0,0],[0,1,0],[0,0,1]])
    best_homography = np.array([[1,0,0],[0,1,0],[0,0,1]])
    if len(target_keypoint) < 6:
        return homography, 0
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
        target_sample = list()
        frame_sample = list()
        for i in range(4):
            idx = random.randint(0, len(target_keypoint)-1)
            target_sample.append(target_keypoint[idx])
            frame_sample.append(frame_keypoint[idx])
        homography = find_homography(target_sample, frame_sample)
        inliers = calculate_inliers(homography, target_keypoint, frame_keypoint,5)
        inlier_rate = inliers / len(target_keypoint)
        if inlier_rate > best_inlier_rate:
            best_inlier_rate = inlier_rate
            best_homography = homography
        if inlier_rate >= 0.9:
            break
        #iteration = math.ceil(math.log(1 - p) / (math.log(1 - best_inlier_rate ** s)+0.0000001))
        e = 1 - best_inlier_rate
        iteration = math.log(1.0 - p) / math.log(1.0 - math.pow(1.0 - e, s))
        iteration_count += 1
    if best_inlier_rate <= 0.4:
        return np.array([[1,0,0],[0,1,0],[0,0,1]]), best_inlier_rate
    return best_homography, best_inlier_rate