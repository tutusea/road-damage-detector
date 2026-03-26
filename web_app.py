"""
YOLO 路面病害检测 - Web 应用
基于 Flask，支持图片上传和实时检测
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import base64
import io
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# 添加当前目录到路径以导入核心模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_core import RoadDamageDetector, CLASS_NAMES

# ==================== 配置参数区 ====================

# Flask 配置
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# 模型配置
MODEL_PATH = "best.pt"
DEVICE = "cpu"  # Render 免费版用 CPU
CONF_THRESHOLD = 0.25

# 上传配置
UPLOAD_FOLDER = "static/temp"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# ==================== Flask 应用 ====================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['JSON_AS_ASCII'] = False

# 全局检测器实例
detector = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_detector():
    """获取检测器实例（单例模式）"""
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH):
            print(f"警告: 模型文件不存在: {MODEL_PATH}")
            return None
        
        print("初始化检测器...")
        detector = RoadDamageDetector(
            model_path=MODEL_PATH,
            device=DEVICE,
            conf_threshold=CONF_THRESHOLD
        )
    return detector


def image_to_base64(image):
    """将 OpenCV 图像转换为 base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def base64_to_image(base64_string):
    """将 base64 字符串转换为 OpenCV 图像"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """检测接口"""
    try:
        detector = get_detector()
        if detector is None:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            }), 500
        
        image = None
        
        # 处理文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': '未选择文件'}), 400
            
            if allowed_file(file.filename):
                # 直接从内存读取图像，避免保存
                filestr = file.read()
                nparr = np.frombuffer(filestr, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'success': False, 'error': '无法读取图片'}), 400
            else:
                return jsonify({'success': False, 'error': '不支持的文件格式'}), 400
        
        # 处理 base64 图像
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image = base64_to_image(image_data)
        
        else:
            return jsonify({'success': False, 'error': '未提供图像'}), 400
        
        # 执行检测
        result = detector.detect(image)
        
        # 转换结果为 base64
        result_image_base64 = image_to_base64(result['result_image'])
        
        # 获取统计信息
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
        print(f"检测错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    detector_status = "ready" if get_detector() is not None else "not_ready"
    return jsonify({
        'status': 'ok',
        'model_status': detector_status,
        'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else None
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """模型信息接口"""
    return jsonify({
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'device': DEVICE,
        'conf_threshold': CONF_THRESHOLD,
        'class_names': CLASS_NAMES
    })


# ==================== 错误处理 ====================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': '文件太大，最大支持 16MB'}), 413


# ==================== 启动应用 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("路面病害检测 Web 服务")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"运行设备: {DEVICE}")
    print("=" * 60)
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    get_detector()
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
