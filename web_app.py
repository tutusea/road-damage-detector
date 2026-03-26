"""
YOLO 路面病害检测 - Web 应用
"""

from flask import Flask, render_template, request, jsonify
import os, sys, base64, cv2, numpy as np

from inference_core import RoadDamageDetector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.25
DEVICE = "cpu"

detector = None
CLASS_NAMES = {0:"D00",1:"D10",2:"D20",3:"D40",4:"D43",5:"D44",6:"D50"}

def get_detector():
    global detector
    if detector is None:
        if not os.path.exists(MODEL_PATH): return None
        detector = RoadDamageDetector(model_path=MODEL_PATH, device=DEVICE, conf_threshold=CONF_THRESHOLD)
    return detector

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        d = get_detector()
        if d is None: return jsonify({'success':False,'error':'Model not loaded'}),500
        image=None
        if 'file' in request.files:
            f=request.files['file']
            nparr=np.frombuffer(f.read(),np.uint8)
            image=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        elif request.json and 'image' in request.json:
            idata=request.json['image']
            if ',' in idata: idata=idata.split(',')[1]
            image=cv2.imdecode(np.frombuffer(base64.b64decode(idata),np.uint8),cv2.IMREAD_COLOR)
        else: return jsonify({'success':False,'error':'No image'}),400
        if image is None: return jsonify({'success':False,'error':'Cannot read'}),400
        result=d.detect(image)
        return jsonify({'success':True,'num_detections':result['num_detections'],'detections':result['detections'],'summary':d.get_detection_summary(result['detections']),'result_image':f'data:image/jpeg;base64,{image_to_base64(result["result_image"])}','class_names':CLASS_NAMES})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/model_info', methods=['GET'])
def model_info(): return jsonify({'model_path':MODEL_PATH,'device':DEVICE,'class_names':CLASS_NAMES})

if __name__=='__main__':
    print("Road Damage Detection Web Service")
    get_detector()
    app.run(host='0.0.0.0',port=5000,debug=True)
