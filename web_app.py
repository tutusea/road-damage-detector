"""
YOLO 路面病害检测 - Web 应用（ONNX 版）
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
import cv2
import numpy as np

# 使用 ONNX 检测器
from inference_onnx import ONNXDetector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

MODEL_PATH = "best.onnx"  # ONNX 模型
CONF_THRESHOLD = 0.25

detector = None

CLASS_NAMES = {
    0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "D43", 5: "D44", 6: "D50"
}


def get_detector():
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model not found: {MODEL_PATH}")
            return None
        try:
            detector = ONNXDetector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)
            print(f"[OK] Detector initialized")
        except Exception as e:
            print(f"Error: {e}")
            return None
    return detector


def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        detector = get_detector()
        if detector is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        image = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            filestr = file.read()
            nparr = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'error': 'Cannot read image'}), 400
        
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        result = detector.detect(image)
        
        result_image_base64 = image_to_base64(result['result_image'])
        summary = detector.get_detection_summary(result['detections'])
        
        return jsonify({
            'success': True,
            'num_detections': result['num_detections'],
            'detections': result['detections'],
            'summary': summary,
            'result_image': f'data:image/jpeg;base64,{result_image_base64}',
            'class_names': CLASS_NAMES
        })
    
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_type': 'ONNX',
        'device': 'CPU',
        'conf_threshold': CONF_THRESHOLD,
        'class_names': CLASS_NAMES
    })


if __name__ == '__main__':
    print("=" * 50)
    print("Road Damage Detection Web Service (ONNX)")
    print("=" * 50)
    get_detector()
    app.run(host='0.0.0.0', port=5000, debug=True)
