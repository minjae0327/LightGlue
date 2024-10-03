from flask import Flask, request, jsonify
import img_process
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        # 이미지 데이터 수신
        image_data = request.data

        if not image_data:
            return jsonify({'result': 'No image data received'}), 400

        # 이미지 디코딩
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'result': 'Invalid image data'}), 400

        # 이미지 처리 (img_process 모듈 사용)
        result = img_process.process_image(img)

        # 결과 반환
        return jsonify({'result': result}), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'result': 'Error processing image', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
