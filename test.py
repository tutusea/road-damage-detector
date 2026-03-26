"""
YOLO测试脚本
在测试集上评估模型性能并保存可视化结果
"""

from ultralytics import YOLO
import os
from datetime import datetime

# ==================== 配置参数 ====================
MODEL_PATH = "models/road_damage_detection/weights/best.pt"  # 模型路径
DATA_YAML = r"G:\ttt\yolo_dataset\data.yaml"  # 数据配置文件路径
TEST_IMAGES_DIR = r"G:\ttt\yolo_dataset\images\test"  # 测试图片目录
OUTPUT_DIR = "outputs/test_results"  # 输出目录
BATCH_SIZE = 8  # 批次大小
IMG_SIZE = 640  # 输入图像大小
DEVICE = "0"  # GPU设备号
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.45  # IoU阈值
SAVE_RESULTS = True  # 是否保存检测结果
SAVE_TXT = True  # 是否保存检测标签
# =================================================

def main():
    """主测试函数"""
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(OUTPUT_DIR, f"test_{timestamp}")
    
    print("=" * 60)
    print("YOLO路面病害检测 - 模型测试")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试图片: {TEST_IMAGES_DIR}")
    print(f"输出目录: {result_dir}")
    print(f"图像大小: {IMG_SIZE}")
    print("=" * 60)
    
    # 检查测试集是否存在
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"\n✗ 测试集不存在: {TEST_IMAGES_DIR}")
        print("请先准备测试集图片")
        return
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(MODEL_PATH)
    
    # 在测试集上进行检测
    print("\n开始测试...")
    results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=SAVE_RESULTS,
        save_txt=SAVE_TXT,
        project=OUTPUT_DIR,
        name=f"test_{timestamp}",
        exist_ok=True,
        verbose=True,
    )
    
    # 统计结果
    total_images = len(results)
    total_detections = sum(len(r.boxes) for r in results)
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"测试图片数: {total_images}")
    print(f"总检测数:   {total_detections}")
    print(f"平均每张图: {total_detections/total_images:.2f} 个目标")
    print(f"\n检测结果已保存至: {result_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
