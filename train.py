"""
YOLO训练脚本 - 适合RTX 3070 8GB
"""

from ultralytics import YOLO
import torch

# ==================== 配置参数 ====================
DATA_YAML = r"G:\ttt\yolo_dataset\data.yaml"  # 数据配置文件路径
MODEL_TYPE = "yolov8n.pt"  # 模型类型: yolov8n.pt(最快), yolov8s.pt, yolov8m.pt, yolov8l.pt
EPOCHS = 100  # 训练轮数
BATCH_SIZE = 8  # 批次大小 (RTX 3070 8GB建议8-16)
IMG_SIZE = 640  # 输入图像大小
DEVICE = "0"  # GPU设备号，"0"表示使用第一块GPU，"cpu"表示使用CPU
WORKERS = 4  # 数据加载线程数
PROJECT = "models"  # 训练结果保存目录
NAME = "road_damage_detection"  # 训练任务名称
PRETRAINED = True  # 是否使用预训练权重
OPTIMIZER = "AdamW"  # 优化器: SGD, Adam, AdamW
LR0 = 0.001  # 初始学习率
LRF = 0.01  # 最终学习率
MOMENTUM = 0.937  # 动量
WEIGHT_DECAY = 0.0005  # 权重衰减
WARMUP_EPOCHS = 3.0  # 预热轮数
# =================================================

def main():
    """主训练函数"""
    
    # 打印训练配置
    print("=" * 60)
    print("YOLO路面病害检测 - 训练配置")
    print("=" * 60)
    print(f"模型类型: {MODEL_TYPE}")
    print(f"数据配置: {DATA_YAML}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"图像大小: {IMG_SIZE}")
    print(f"训练设备: {'GPU ' + DEVICE if DEVICE != 'cpu' else 'CPU'}")
    print(f"优化器: {OPTIMIZER}")
    print(f"初始学习率: {LR0}")
    print(f"结果保存: {PROJECT}/{NAME}")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(MODEL_TYPE)
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        pretrained=PRETRAINED,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        verbose=True,
        seed=42,  # 随机种子，保证可重复
        patience=50,  # 早停耐心值
        save=True,  # 保存检查点
        save_period=10,  # 每10轮保存一次
        plots=True,  # 生成训练图表
    )
    
    # 打印训练结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳模型: {PROJECT}/{NAME}/weights/best.pt")
    print(f"最后模型: {PROJECT}/{NAME}/weights/last.pt")
    print(f"训练结果: {PROJECT}/{NAME}/results.png")
    print(f"混淆矩阵: {PROJECT}/{NAME}/confusion_matrix.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
