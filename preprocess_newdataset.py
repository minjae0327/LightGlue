import os
import json
import copy

aircraft_datasets = "D:/AMARA/"

labels = os.path.join(aircraft_datasets + "label/")
video_dir = os.path.join(aircraft_datasets, "video/")
output_dir = os.path.join(aircraft_datasets, "frames_from_video/")
target_image_dir = os.path.join(aircraft_datasets, "image/")

def get_origin_coordinates():
    # 배열 초기화
    origin_coordinate_1 = []
    origin_coordinate_2 = []
    origin_coordinate_3 = []
    origin_coordinate_4 = []
    origin_coordinate_5 = []
    float_origin_coordinate_1 = []
    float_origin_coordinate_2 = []
    float_origin_coordinate_3 = []
    float_origin_coordinate_4 = []
    float_origin_coordinate_5 = []

    # 배열 리스트 생성
    origin_coordinates = [
        origin_coordinate_1,
        origin_coordinate_2,
        origin_coordinate_3,
        origin_coordinate_4,
        origin_coordinate_5
    ]

    float_origin_coordinates = [
        float_origin_coordinate_1,
        float_origin_coordinate_2,
        float_origin_coordinate_3,
        float_origin_coordinate_4,
        float_origin_coordinate_5
    ]

    # 원점 좌표값 불러오기
    label_index = 0
    for label in os.listdir(labels):
        label_path = os.path.join(labels, label)
        for label_file in os.listdir(label_path):
            with open(os.path.join(label_path, label_file), "r") as f:
                json_file = json.load(f)
                coord = json_file["annotationList"]
                _coord = copy.deepcopy(coord)
                _coord = _coord[1:-1]
                
                # 현재 반복에 해당하는 배열에 요소 추가
                float_origin_coordinates[label_index].append(_coord)
                
                for i in range(len(coord)):
                    coord[i][0] = coord[i][0] * 640
                    coord[i][1] = coord[i][1] * 480
                coord = coord[1:-1]
                origin_coordinates[label_index].append(coord)
        
        label_index += 1
        if label_index >= 5:
            break
        
    
    return origin_coordinates, float_origin_coordinates

def get_lens():
    # 원본 이미지 경로를 저장할 리스트
    len_1 = []
    len_2 = []
    len_3 = []
    len_4 = []
    len_5 = []

    lens = [
        len_1,
        len_2,
        len_3,
        len_4,
        len_5
    ]

    i = 0

    index = 0
    for videos in os.listdir(output_dir):
        image_path = os.path.join(output_dir, videos)
        for image_file in os.listdir(image_path):
            _path = os.path.join(image_path, image_file)
            for image in os.listdir(_path):
                lens[index].append(os.path.join(_path, image))
        
        index += 1
        if index >= 5:
            break

    return lens


def get_images():
    # 원본 이미지 경로를 저장할 리스트
    image_1 = []
    image_2 = []
    image_3 = []
    image_4 = []
    image_5 = []

    images = [
        image_1,
        image_2,
        image_3,
        image_4,
        image_5
    ]

    i = 0

    index = 0
    for videos in os.listdir(output_dir):
        image_path = os.path.join(output_dir, videos)
        for image_file in os.listdir(image_path):
            _path = os.path.join(image_path, image_file)
            list = []
            for image in os.listdir(_path):
                list.append(os.path.join(_path, image))
                # 현재 반복에 해당하는 배열에 요소 추가
            images[index].append(list)
        
        index += 1
        if index >= 5:
            break

    # images 리스트의 길이 반환
    # num_images = len(images)
    # print(f"총 이미지 수: {num_images}")