from flask import Flask, request, jsonify
import img_process

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST', 'PUT'])
def upload_frame():
    image_data = request.data
    
    # 이미지 전처리
    # image = img_process.preprocess_image(image_data)
    
    # 예시: 처리된 결과를 result로 설정 (실제 처리 로직에 맞게 수정)
    result = "Frame received and processed"
    
    # JSON 형식으로 응답 반환
    response = jsonify({'result': result})
    print(f"Sending response: {response.get_data(as_text=True)}")  # 응답 내용 로그 출력
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
