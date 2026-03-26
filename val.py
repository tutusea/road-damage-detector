"""
YOLO验证脚本
在验证集上评估模型性能
"""

from ultralytics import YOLO

# ==================== 配置参数 ====================
MODEL_PATH = "models/road_damage_detection/weights/best.pt"  # 模型路径
DATA_YAML = r"G:\ttt\yolo_dataset\data.yaml"  # 数据配置文件路径
BATCH_SIZE = 8  # 批次大小
IMG_SIZE = 640  # 输入图像大小
DEVICE = "0"  # GPU设备号
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.45  # IoU阈值
SAVE_RESULTS = True  # 是否保存验证结果
# =================================================

def main():
    """主验证函数"""
    
    print("=" * 60)
    print("YOLO路面病害检测 - 模型验证")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据配置: {DATA_YAML}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"图像大小: {IMG_SIZE}")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(MODEL_PATH)
    
    # 验证模型
    print("\n开始验证...")
    metrics = model.val(
        data=DATA_YAML,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=SAVE_RESULTS,
        save_json=False,
        verbose=True,
    )
    
    # 打印验证结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"F1-Score:   {2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-16):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
