"""
YOLO 推理脚本
支持单张图片和批量检测
"""

import os
import sys
import argparse
from pathlib import Path
from inference_core import RoadDamageDetector, detect_image, detect_folder
import json
import cv2

# ==================== 配置参数区 ====================

# 默认模型路径（训练完成后更新）
DEFAULT_MODEL_PATH = "runs/detect/models/road_damage_detection2/weights/best.pt"

# 默认置信度阈值
DEFAULT_CONF_THRESHOLD = 0.25

# 默认设备（0=GPU, cpu=CPU）
DEFAULT_DEVICE = "0"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "outputs/inference"

# ==================== 推理代码 ====================

def inference_single(image_path: str,
                    model_path: str = DEFAULT_MODEL_PATH,
                    device: str = DEFAULT_DEVICE,
                    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                    save_result: bool = True,
                    output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    单张图片推理
    """
    # 检查文件
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        print("请先完成训练或检查模型路径")
        return
    
    print(f"\n开始单张图片检测")
    print(f"图片: {image_path}")
    print(f"模型: {model_path}")
    
    # 确定保存路径
    save_path = None
    if save_result:
        img_name = Path(image_path).stem
        save_path = os.path.join(output_dir, f"{img_name}_result.jpg")
    
    # 执行检测
    result = detect_image(
        image_path=image_path,
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold,
        save_result=save_result,
        save_path=save_path
    )
    
    # 输出结果
    print(f"\n检测完成！")
    print(f"发现 {result['num_detections']} 个目标")
    
    if result['num_detections'] > 0:
        print("\n检测详情:")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']} (置信度: {det['confidence']:.2%})")
            print(f"     位置: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) - "
                  f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
    
    if save_result:
        print(f"\n结果图保存至: {save_path}")
        
        # 显示结果（如果环境支持）
        try:
            result_img = result['result_image']
            cv2.imshow("Detection Result", result_img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    
    return result


def inference_folder(folder_path: str,
                    model_path: str = DEFAULT_MODEL_PATH,
                    device: str = DEFAULT_DEVICE,
                    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                    output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    批量文件夹推理
    """
    # 检查路径
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        print("请先完成训练或检查模型路径")
        return
    
    print(f"\n开始批量检测")
    print(f"文件夹: {folder_path}")
    print(f"模型: {model_path}")
    print(f"输出目录: {output_dir}")
    
    # 执行批量检测
    results = detect_folder(
        folder_path=folder_path,
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold,
        output_dir=output_dir
    )
    
    # 生成统计报告
    total_images = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    total_detections = sum(r.get('num_detections', 0) for r in results)
    
    print(f"\n批量检测完成！")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {successful}")
    print(f"失败: {total_images - successful}")
    print(f"总检测数: {total_detections}")
    
    # 保存详细报告
    report_path = os.path.join(output_dir, "inference_report.json")
    os.makedirs(output_dir, exist_ok=True)
    
    # 简化结果用于保存（移除图像数据）
    report_data = []
    for r in results:
        report_item = {
            "image_path": r.get("image_path"),
            "success": r.get("success", False),
            "num_detections": r.get("num_detections", 0),
            "detections": r.get("detections", []),
        }
        if "error" in r:
            report_item["error"] = r["error"]
        report_data.append(report_item)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"详细报告保存至: {report_path}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO 路面病害检测推理")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["single", "folder"],
                       help="推理模式: single=单张图片, folder=整个文件夹")
    parser.add_argument("--input", type=str, required=True,
                       help="输入路径（图片或文件夹）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                       help=f"模型路径 (默认: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                       help=f"运行设备 (默认: {DEFAULT_DEVICE})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                       help=f"置信度阈值 (默认: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存结果图")
    
    args = parser.parse_args()
    
    # 根据模式执行推理
    if args.mode == "single":
        inference_single(
            image_path=args.input,
            model_path=args.model,
            device=args.device,
            conf_threshold=args.conf,
            save_result=not args.no_save,
            output_dir=args.output
        )
    elif args.mode == "folder":
        inference_folder(
            folder_path=args.input,
            model_path=args.model,
            device=args.device,
            conf_threshold=args.conf,
            output_dir=args.output
        )


if __name__ == "__main__":
    # 如果没有命令行参数，显示使用示例
    if len(sys.argv) == 1:
        print("YOLO 路面病害检测推理脚本")
        print("=" * 50)
        print("\n使用示例:")
        print("\n1. 单张图片检测:")
        print("   python inference.py --mode single --input path/to/image.jpg")
        print("\n2. 批量文件夹检测:")
        print("   python inference.py --mode folder --input path/to/folder")
        print("\n3. 自定义参数:")
        print("   python inference.py --mode single --input image.jpg --conf 0.3 --device cpu")
        print("\n完整参数说明:")
        print("   python inference.py --help")
        print("\n")
        
        # 询问是否继续
        response = input("是否运行测试示例? (y/n): ")
        if response.lower() == 'y':
            # 运行示例
            test_image = input("请输入测试图片路径: ")
            if test_image and os.path.exists(test_image):
                inference_single(test_image)
    else:
        main()
