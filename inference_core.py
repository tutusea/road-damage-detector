"""
推理核心模块 - 网页版和本地版共用
封装YOLO模型的加载和推理逻辑
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Union, List, Dict, Tuple
import time

# 类别名称映射（对应 data.yaml）
CLASS_NAMES = {
    0: "D00",   # 纵向裂缝
    1: "D10",   # 横向裂缝
    2: "D20",   # 网状裂缝
    3: "D40",   # 坑槽
    4: "D43",   # 松散
    5: "D44",   # 车辙
    6: "D50",   # 修补
}

# 类别颜色映射（BGR格式用于OpenCV）
CLASS_COLORS = {
    0: (0, 0, 255),     # D00 - 红色
    1: (0, 255, 0),     # D10 - 绿色
    2: (255, 0, 0),     # D20 - 蓝色
    3: (0, 255, 255),   # D40 - 黄色
    4: (255, 0, 255),   # D43 - 紫色
    5: (255, 255, 0),   # D44 - 青色
    6: (128, 128, 128), # D50 - 灰色
}


# ==================== 模型路径自动检测 ====================
import sys

def get_default_model_path():
    """自动检测模型路径，支持打包后的 exe"""
    import os
    
    # 打包后的 exe 同目录
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    
    # 可能的模型路径列表（按优先级排序）
    possible_paths = [
        # 打包后的 exe 同目录（直接放 best.pt）
        os.path.join(exe_dir, "best.pt"),
        # 打包后的 weights 文件夹
        os.path.join(exe_dir, "weights", "best.pt"),
        # 开发环境路径
        os.path.abspath("runs/detect/models/road_damage_detection2/weights/best.pt"),
        os.path.abspath("models/road_damage_detection/weights/best.pt"),
    ]
    
    for path in possible_paths:
        # 处理相对路径
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        if os.path.exists(path):
            print(f"找到模型: {path}")
            return path
    
    # 返回第一个路径作为默认
    print(f"警告: 未找到模型，使用默认路径: {possible_paths[-1]}")
    return possible_paths[-1]


class RoadDamageDetector:
    """路面病害检测器 - 网页版和本地版共用"""
    
    def __init__(self, model_path: str = None, device: str = "0", conf_threshold: float = 0.25):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，默认使用训练好的模型
            device: 计算设备，"0"表示GPU，"cpu"表示CPU
            conf_threshold: 置信度阈值
        """
        if model_path is None:
            # 自动检测模型路径
            model_path = get_default_model_path()
        
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = []
        
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # 获取类别名称
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = CLASS_NAMES
        
        print(f"✓ 模型加载成功")
        print(f"  类别数: {len(self.class_names)}")
        print(f"  设备: {'GPU' if self.device != 'cpu' else 'CPU'}")
    
    def detect(self, image: Union[str, np.ndarray], conf_threshold: float = None) -> Dict:
        """
        对单张图片进行检测
        
        Args:
            image: 图片路径或numpy数组
            conf_threshold: 置信度阈值（覆盖默认值）
        
        Returns:
            包含检测结果的字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # 记录推理时间
        start_time = time.time()
        
        # 执行推理
        results = self.model.predict(
            source=image,
            conf=conf,
            device=self.device,
            verbose=False
        )[0]
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 解析结果
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # 边界框坐标
            confidences = results.boxes.conf.cpu().numpy()  # 置信度
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # 类别ID
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                detection = {
                    'id': i,
                    'class_id': int(cls_id),
                    'class_name': self.class_names.get(cls_id, f"class_{cls_id}") if isinstance(self.class_names, dict) else self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1]),
                        'center_x': float((box[0] + box[2]) / 2),
                        'center_y': float((box[1] + box[3]) / 2),
                    }
                }
                detections.append(detection)
        
        # 生成可视化结果图
        annotated_image = results.plot() if hasattr(results, 'plot') else None
        
        # 获取统计信息
        summary = self.get_detection_summary(detections)
        
        return {
            'success': True,
            'num_detections': len(detections),
            'detections': detections,
            'inference_time_ms': inference_time,
            'result_image': annotated_image,
            'annotated_image': annotated_image,
            'class_names': self.class_names,
            'summary': summary,
        }
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        获取检测结果统计摘要
        
        Args:
            detections: 检测结果列表
        
        Returns:
            统计摘要字典
        """
        summary = {
            "total_detections": len(detections),
            "by_class": {},
            "average_confidence": 0.0,
        }
        
        if not detections:
            return summary
        
        # 按类别统计
        class_counts = {}
        confidences = []
        
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det["confidence"])
        
        summary["by_class"] = class_counts
        summary["average_confidence"] = sum(confidences) / len(confidences)
        
        return summary
    
    def detect_batch(self, images: List[Union[str, np.ndarray]], conf_threshold: float = None) -> List[Dict]:
        """
        批量检测多张图片
        
        Args:
            images: 图片路径或numpy数组列表
            conf_threshold: 置信度阈值
        
        Returns:
            检测结果列表
        """
        results = []
        for img in images:
            result = self.detect(img, conf_threshold)
            results.append(result)
        return results
    
    def get_class_names(self) -> Dict:
        """获取类别名称字典"""
        return CLASS_NAMES
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
        }


class YOLODetector(RoadDamageDetector):
    """保持向后兼容的别名"""
    pass


def draw_detections(image: np.ndarray, detections: List[Dict], color_map: Dict = None) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 原始图像
        detections: 检测结果列表
        color_map: 类别颜色映射
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    if color_map is None:
        # 使用预定义的颜色
        color_map = CLASS_COLORS
    
    for det in detections:
        cls_id = det['class_id']
        cls_name = det['class_name']
        conf = det['confidence']
        bbox = det['bbox']
        
        color = color_map.get(cls_id, (0, 255, 0))
        
        # 绘制边界框
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        
        cv2.rectangle(img, (x1, label_y - label_size[1] - 4), 
                     (x1 + label_size[0], label_y), color, -1)
        cv2.putText(img, label, (x1, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img


def format_detection_info(detections: List[Dict]) -> str:
    """格式化检测信息为文本"""
    if len(detections) == 0:
        return "未检测到目标"
    
    lines = [f"检测到 {len(detections)} 个目标:", ""]
    
    # 按类别统计
    class_count = {}
    for det in detections:
        cls_name = det['class_name']
        class_count[cls_name] = class_count.get(cls_name, 0) + 1
    
    lines.append("【类别统计】")
    for cls_name, count in class_count.items():
        lines.append(f"  {cls_name}: {count} 个")
    
    lines.append("")
    lines.append("【详细信息】")
    for det in detections:
        bbox = det['bbox']
        lines.append(f"  ID {det['id']}: {det['class_name']} "
                    f"(置信度: {det['confidence']:.2%}, "
                    f"位置: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}])")
    
    return "\n".join(lines)


# 便捷函数
def create_detector(model_path: str = None, device: str = "0", conf_threshold: float = 0.25):
    """创建检测器实例的便捷函数"""
    return RoadDamageDetector(model_path, device, conf_threshold)


def detect_image(image_path: str, model_path: str = None, device: str = "0", conf_threshold: float = 0.25):
    """单张图片检测的便捷函数"""
    detector = create_detector(model_path, device, conf_threshold)
    return detector.detect(image_path)


# 测试代码
if __name__ == "__main__":
    # 测试检测器
    print("=" * 60)
    print("测试推理核心模块")
    print("=" * 60)
    
    # 创建检测器实例（如果模型存在）
    model_path = "models/road_damage_detection/weights/best.pt"
    if os.path.exists(model_path):
        detector = RoadDamageDetector(model_path)
        
        # 打印模型信息
        info = detector.get_model_info()
        print("\n模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n模型不存在: {model_path}")
        print("请先训练模型")
