from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST', 'PUT'])
def upload_frame():
    image_data = request.data
    npimg = np.frombuffer(image_data, np.uint8)  # 바이트 데이터를 numpy 배열로 변환
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # numpy 배열을 이미지로 디코딩
    
    # 이미지 저장 (OpenCV 사용)
    cv2.imwrite('received_frame.png', image)

    # 여기서 추가적인 이미지 처리를 수행할 수 있습니다.
    
    return "Frame received", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
