"""
轻量级 ONNX 推理模块 - 内存占用小，适合 Render 免费版
"""

import numpy as np
import cv2
import onnxruntime as ort
import os
from typing import Dict, List

# 类别名称
CLASS_NAMES = {
    0: "D00", 1: "D10", 2: "D20", 3: "D40", 4: "D43", 5: "D44", 6: "D50"
}

CLASS_COLORS = {
    0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0),
    3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0), 6: (128, 128, 128)
}

CLASS_DESCRIPTIONS = {
    "D00": "纵向裂缝", "D10": "横向裂缝", "D20": "网状裂缝",
    "D40": "坑槽", "D43": "松散", "D44": "车辙", "D50": "修补"
}


class ONNXDetector:
    """轻量级 ONNX 检测器"""
    
    def __init__(self, model_path="best.onnx", conf_threshold=0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.conf_threshold = conf_threshold
        self.img_size = 640
        self.model_path = model_path
        
        # 创建 session（CPU only，更省内存）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # 限制线程数
        sess_options.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"✓ ONNX 模型加载成功: {model_path}")
    
    def detect(self, image, conf_threshold=None):
        """检测图像"""
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        # 原始图像尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # 后处理
        detections = self._postprocess(outputs, orig_w, orig_h, conf_threshold)
        
        # 绘制结果
        result_img = self._draw_detections(image.copy(), detections)
        
        return {
            'success': True,
            'num_detections': len(detections),
            'detections': detections,
            'result_image': result_img
        }
    
    def _postprocess(self, outputs, orig_w, orig_h, conf_threshold):
        """后处理 - 解析 YOLO 输出"""
        detections = []
        
        # 解析输出 (1, num_boxes, 85) = (1, 8400, 85)
        predictions = outputs[0][0]
        
        # 遍历所有预测框
        for pred in predictions:
            # pred[4] 是 objectness score
            obj_score = pred[4]
            if obj_score < conf_threshold:
                continue
            
            # 获取类别分数
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # 最终置信度
            final_conf = obj_score * class_score
            if final_conf < conf_threshold:
                continue
            
            # 边界框 (中心x, 中心y, 宽, 高) -> 归一化到图像尺寸
            cx, cy, w, h = pred[:4]
            
            # 转换为像素坐标
            x1 = int((cx - w/2) * orig_w)
            y1 = int((cy - h/2) * orig_h)
            x2 = int((cx + w/2) * orig_w)
            y2 = int((cy + h/2) * orig_h)
            
            # 边界检查
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            class_name = CLASS_NAMES.get(int(class_id), f"class_{class_id}")
            
            detections.append({
                'id': len(detections),
                'class_id': int(class_id),
                'class_name': class_name,
                'confidence': float(final_conf),
                'bbox': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1,
                    'center_x': (x1 + x2) / 2, 'center_y': (y1 + y2) / 2
                }
            })
        
        return detections
    
    def _draw_detections(self, img, detections):
        """绘制检测框"""
        for det in detections:
            x1 = int(det['bbox']['x1'])
            y1 = int(det['bbox']['y1'])
            x2 = int(det['bbox']['x2'])
            y2 = int(det['bbox']['y2'])
            
            color = CLASS_COLORS.get(det['class_id'], (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 标签背景
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """获取检测统计摘要"""
        summary = {
            "total_detections": len(detections),
            "by_class": {},
            "average_confidence": 0.0
        }
        
        if not detections:
            return summary
        
        class_counts = {}
        confidences = []
        
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det['confidence'])
        
        summary["by_class"] = class_counts
        summary["average_confidence"] = sum(confidences) / len(confidences)
        
        return summary
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'model_type': 'ONNX',
            'conf_threshold': self.conf_threshold,
            'num_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES
        }


# 便捷函数
def create_detector(model_path="best.onnx", conf_threshold=0.25):
    """创建检测器实例"""
    return ONNXDetector(model_path, conf_threshold)


# 测试
if __name__ == "__main__":
    print("=" * 50)
    print("测试 ONNX 检测器")
    print("=" * 50)
    
    if os.path.exists("best.onnx"):
        detector = ONNXDetector("best.onnx")
        print(f"模型信息: {detector.get_model_info()}")
    else:
        print("模型文件不存在: best.onnx")