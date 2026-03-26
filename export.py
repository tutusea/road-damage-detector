"""
YOLO 模型导出脚本
支持导出为 ONNX, TensorRT, OpenVINO, CoreML, TensorFlow 等格式
"""

from ultralytics import YOLO
import os

# ==================== 配置参数区 ====================

# 模型路径（训练完成后更新此路径）
MODEL_PATH = "runs/detect/models/road_damage_detection2/weights/best.pt"

# 导出格式选项
# 可选: "onnx", "engine" (TensorRT), "openvino", "coreml", "tflite", "pb", "saved_model", "torchscript"
EXPORT_FORMAT = "onnx"

# 导出配置
IMG_SIZE = 640  # 输入图像尺寸
HALF = False    # 是否使用 FP16 半精度（加速推理，减少显存）
INT8 = False    # 是否使用 INT8 量化（进一步加速，可能有精度损失）
SIMPLIFY = True # 是否简化 ONNX 模型（推荐）
OPSET = 12      # ONNX opset 版本

# 输出路径（可选，默认与模型同目录）
OUTPUT_PATH = None  # 例如: "models/export/best.onnx"

# ==================== 导出代码 ====================

def export_model():
    """导出模型"""
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请确认训练已完成或修改 MODEL_PATH 路径")
        return
    
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"\n开始导出...")
    print(f"格式: {EXPORT_FORMAT}")
    print(f"图像尺寸: {IMG_SIZE}")
    print(f"FP16半精度: {HALF}")
    print(f"INT8量化: {INT8}")
    
    # 导出模型
    try:
        model.export(
            format=EXPORT_FORMAT,
            imgsz=IMG_SIZE,
            half=HALF,
            int8=INT8,
            simplify=SIMPLIFY,
            opset=OPSET,
        )
        
        print(f"\n导出成功！")
        
        # 显示导出文件路径
        model_dir = os.path.dirname(MODEL_PATH)
        model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
        
        format_extensions = {
            "onnx": ".onnx",
            "engine": ".engine",
            "openvino": "_openvino_model",
            "coreml": ".mlmodel",
            "tflite": ".tflite",
            "pb": "",
            "saved_model": "_saved_model",
            "torchscript": ".torchscript",
        }
        
        ext = format_extensions.get(EXPORT_FORMAT, f".{EXPORT_FORMAT}")
        if OUTPUT_PATH:
            exported_path = OUTPUT_PATH
        else:
            exported_path = os.path.join(model_dir, f"{model_name}{ext}")
        
        print(f"导出文件: {exported_path}")
        
        if os.path.exists(exported_path):
            size_mb = os.path.getsize(exported_path) / (1024 * 1024)
            print(f"文件大小: {size_mb:.2f} MB")
        
        print(f"\n提示:")
        if EXPORT_FORMAT == "onnx":
            print("- ONNX 模型可用于多种推理框架")
            print("- 可使用 Netron 查看模型结构: https://netron.app")
        elif EXPORT_FORMAT == "engine":
            print("- TensorRT 模型仅支持 NVIDIA GPU")
            print("- 推理速度最快，但导出需要较长时间")
        elif EXPORT_FORMAT == "openvino":
            print("- OpenVINO 模型适合 Intel CPU/GPU")
            print("- 在 Intel 硬件上推理性能优秀")
        elif EXPORT_FORMAT == "tflite":
            print("- TFLite 模型适合移动端和嵌入式设备")
            print("- 文件体积小，推理速度快")
        
        return exported_path
        
    except Exception as e:
        print(f"\n导出失败: {str(e)}")
        print("\n常见错误:")
        print("- TensorRT 导出需要安装 tensorrt 包")
        print("- OpenVINO 导出需要安装 openvino 包")
        print("- CoreML 导出仅支持 macOS")
        return None


def export_all_formats():
    """导出所有可用格式"""
    
    formats_to_try = [
        ("onnx", "ONNX 格式（通用）"),
        ("torchscript", "TorchScript 格式（PyTorch）"),
    ]
    
    # 可选格式（根据平台）
    optional_formats = [
        ("engine", "TensorRT 格式（NVIDIA GPU）"),
        ("openvino", "OpenVINO 格式（Intel）"),
    ]
    
    print("=" * 60)
    print("批量导出模型到多种格式")
    print("=" * 60)
    
    results = {}
    
    # 尝试导出必需格式
    for fmt, desc in formats_to_try:
        print(f"\n尝试导出: {desc}")
        global EXPORT_FORMAT
        EXPORT_FORMAT = fmt
        path = export_model()
        results[fmt] = path
    
    # 询问是否导出可选格式
    print("\n" + "=" * 60)
    response = input("是否继续导出其他格式? (y/n): ")
    
    if response.lower() == 'y':
        for fmt, desc in optional_formats:
            print(f"\n尝试导出: {desc}")
            EXPORT_FORMAT = fmt
            path = export_model()
            results[fmt] = path
    
    # 输出总结
    print("\n" + "=" * 60)
    print("导出总结:")
    print("=" * 60)
    for fmt, path in results.items():
        status = "成功" if path else "失败"
        print(f"{fmt}: {status}")
        if path:
            print(f"  路径: {path}")


def test_exported_model(model_path: str):
    """测试导出的模型"""
    print(f"\n测试模型: {model_path}")
    
    try:
        # 加载导出的模型
        model = YOLO(model_path)
        
        # 测试推理
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_img, verbose=False)
        
        print("模型测试成功！")
        return True
        
    except Exception as e:
        print(f"模型测试失败: {str(e)}")
        return False


# ==================== 主函数 ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            # 导出所有格式
            export_all_formats()
        elif command == "test" and len(sys.argv) > 2:
            # 测试指定模型
            test_exported_model(sys.argv[2])
        else:
            # 导出指定格式
            EXPORT_FORMAT = command
            export_model()
    else:
        # 默认导出配置中的格式
        print("YOLO 模型导出脚本")
        print("=" * 50)
        print("\n使用方式:")
        print("  python export.py           # 导出默认格式 (ONNX)")
        print("  python export.py onnx      # 导出 ONNX")
        print("  python export.py engine    # 导出 TensorRT")
        print("  python export.py openvino  # 导出 OpenVINO")
        print("  python export.py all       # 批量导出所有格式")
        print("  python export.py test <path> # 测试导出模型")
        print("\n当前配置:")
        print(f"  格式: {EXPORT_FORMAT}")
        print(f"  模型: {MODEL_PATH}")
        print(f"  尺寸: {IMG_SIZE}")
        print("=" * 50)
        
        export_model()
