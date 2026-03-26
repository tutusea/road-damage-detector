"""
YOLO 路面病害检测 - 桌面应用
基于 tkinter，支持选择图片并显示检测结果
可以打包成 exe 用于课堂演示
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加当前目录到路径以导入核心模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_core import RoadDamageDetector, CLASS_NAMES, CLASS_COLORS

# ==================== 模型路径自动检测 ====================

def get_model_path():
    """自动检测模型路径，支持打包后的 exe"""
    # 开发环境下的默认路径
    dev_path = "runs/detect/models/road_damage_detection2/weights/best.pt"
    
    # 打包后的 exe 同目录
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        frozen_path = os.path.join(base_dir, "weights", "best.pt")
        if os.path.exists(frozen_path):
            return frozen_path
    
    # 开发环境
    if os.path.exists(dev_path):
        return dev_path
    
    # 返回默认路径
    return dev_path

# 模型配置
MODEL_PATH = get_model_path()
DEVICE = "0"  # "0"=GPU, "cpu"=CPU
CONF_THRESHOLD = 0.25

# 窗口配置
WINDOW_TITLE = "路面病害检测系统"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# 图像显示尺寸
MAX_IMAGE_WIDTH = 800
MAX_IMAGE_HEIGHT = 600

# ==================== 桌面应用类 ====================

class RoadDamageDetectionApp:
    """路面病害检测桌面应用"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(1000, 700)
        
        # 初始化变量
        self.detector = None
        self.current_image = None
        self.current_image_path = None
        self.detection_result = None
        
        # 创建界面
        self.create_widgets()
        
        # 加载模型
        self.load_model()
    
    def create_widgets(self):
        """创建界面组件"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding="10")
        toolbar.pack(fill='x')
        
        ttk.Button(toolbar, text="选择图片", command=self.select_image).pack(side='left', padx=5)
        ttk.Button(toolbar, text="重新检测", command=self.detect_current).pack(side='left', padx=5)
        ttk.Button(toolbar, text="保存结果", command=self.save_result).pack(side='left', padx=5)
        
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=10)
        
        ttk.Label(toolbar, text="置信度阈值:").pack(side='left')
        self.conf_var = tk.DoubleVar(value=CONF_THRESHOLD)
        conf_spinbox = ttk.Spinbox(toolbar, from_=0.1, to=0.9, increment=0.05, 
                                   textvariable=self.conf_var, width=5)
        conf_spinbox.pack(side='left', padx=5)
        
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=10)
        
        self.model_status_var = tk.StringVar(value="模型未加载")
        ttk.Label(toolbar, textvariable=self.model_status_var).pack(side='left')
        
        # 主内容区
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 左侧：图片显示
        image_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="10")
        image_frame.pack(side='left', fill='both', expand=True)
        
        # 创建标签页
        self.notebook = ttk.Notebook(image_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # 原图标签页
        self.original_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.original_tab, text="原图")
        self.original_label = ttk.Label(self.original_tab)
        self.original_label.pack(fill='both', expand=True)
        
        # 结果图标签页
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="检测结果")
        self.result_label = ttk.Label(self.result_tab)
        self.result_label.pack(fill='both', expand=True)
        
        # 右侧：信息面板
        info_frame = ttk.Frame(main_frame, width=350)
        info_frame.pack(side='right', fill='y', padx=(10, 0))
        info_frame.pack_propagate(False)
        
        # 检测统计
        stats_frame = ttk.LabelFrame(info_frame, text="检测统计", padding="10")
        stats_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, wrap='word', state='disabled')
        self.stats_text.pack(fill='both', expand=True)
        
        # 检测详情
        details_frame = ttk.LabelFrame(info_frame, text="检测详情", padding="10")
        details_frame.pack(fill='both', expand=True)
        
        # 创建树形控件
        columns = ('ID', '类别', '置信度', '位置')
        self.tree = ttk.Treeview(details_frame, columns=columns, show='headings', height=15)
        
        self.tree.heading('ID', text='ID')
        self.tree.heading('类别', text='类别')
        self.tree.heading('置信度', text='置信度')
        self.tree.heading('位置', text='位置')
        
        self.tree.column('ID', width=40)
        self.tree.column('类别', width=80)
        self.tree.column('置信度', width=80)
        self.tree.column('位置', width=120)
        
        scrollbar = ttk.Scrollbar(details_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(fill='x', side='bottom')
    
    def load_model(self):
        """加载检测模型"""
        try:
            if not os.path.exists(MODEL_PATH):
                self.model_status_var.set(f"模型不存在: {MODEL_PATH}")
                messagebox.showwarning("警告", f"模型文件不存在:\n{MODEL_PATH}\n\n请先完成训练")
                return
            
            self.status_var.set("正在加载模型...")
            self.root.update()
            
            self.detector = RoadDamageDetector(
                model_path=MODEL_PATH,
                device=DEVICE,
                conf_threshold=self.conf_var.get()
            )
            
            self.model_status_var.set(f"模型已加载 | 设备: {DEVICE}")
            self.status_var.set("模型加载完成")
            
        except Exception as e:
            self.model_status_var.set("模型加载失败")
            self.status_var.set(f"错误: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")
    
    def select_image(self):
        """选择图片文件"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
            ("所有文件", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="选择图片",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_image_path = filepath
            self.load_and_display_image(filepath)
            self.detect_current()
    
    def load_and_display_image(self, filepath):
        """加载并显示原图"""
        try:
            # 读取图像
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("无法读取图像")
            
            self.current_image = image
            
            # 显示原图
            self.display_image_on_label(image, self.original_label)
            
            self.status_var.set(f"已加载: {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败:\n{str(e)}")
    
    def display_image_on_label(self, image, label, max_width=MAX_IMAGE_WIDTH, max_height=MAX_IMAGE_HEIGHT):
        """在标签上显示图像"""
        # 转换颜色空间 (BGR -> RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # 缩放图像
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # 转换为 PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # 显示
        label.configure(image=photo)
        label.image = photo  # 保持引用
    
    def detect_current(self):
        """对当前图片进行检测"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        if self.detector is None:
            messagebox.showwarning("警告", "模型未加载")
            return
        
        try:
            self.status_var.set("正在检测...")
            self.root.update()
            
            # 更新置信度阈值
            self.detector.conf_threshold = self.conf_var.get()
            
            # 执行检测
            result = self.detector.detect(self.current_image)
            self.detection_result = result
            
            # 显示结果图
            result_image = result['result_image']
            self.display_image_on_label(result_image, self.result_label)
            
            # 更新统计信息
            self.update_stats(result)
            
            # 更新详情表格
            self.update_details(result['detections'])
            
            # 切换到结果标签页
            self.notebook.select(self.result_tab)
            
            self.status_var.set(f"检测完成，发现 {result['num_detections']} 个目标")
            
        except Exception as e:
            self.status_var.set(f"检测失败: {str(e)}")
            messagebox.showerror("错误", f"检测失败:\n{str(e)}")
    
    def update_stats(self, result):
        """更新统计信息"""
        self.stats_text.configure(state='normal')
        self.stats_text.delete('1.0', 'end')
        
        num = result['num_detections']
        self.stats_text.insert('end', f"检测目标总数: {num}\n")
        self.stats_text.insert('end', "=" * 30 + "\n\n")
        
        if num > 0:
            summary = self.detector.get_detection_summary(result['detections'])
            self.stats_text.insert('end', "按类别统计:\n")
            for class_name, count in summary['by_class'].items():
                class_desc = {
                    'D00': '纵向裂缝',
                    'D10': '横向裂缝',
                    'D20': '网状裂缝',
                    'D40': '坑槽',
                    'D43': '松散',
                    'D44': '车辙',
                    'D50': '修补'
                }
                desc = class_desc.get(class_name, class_name)
                self.stats_text.insert('end', f"  {class_name} ({desc}): {count}\n")
            
            self.stats_text.insert('end', f"\n平均置信度: {summary['average_confidence']:.2%}\n")
        else:
            self.stats_text.insert('end', "未检测到病害\n")
        
        self.stats_text.configure(state='disabled')
    
    def update_details(self, detections):
        """更新检测详情表格"""
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 填充数据
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            position = f"({bbox['x1']:.0f}, {bbox['y1']:.0f})"
            
            self.tree.insert('', 'end', values=(
                i,
                det['class_name'],
                f"{det['confidence']:.2%}",
                position
            ))
    
    def save_result(self):
        """保存检测结果"""
        if self.detection_result is None:
            messagebox.showwarning("警告", "没有检测结果可保存")
            return
        
        filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有文件", "*.*")]
        
        default_name = ""
        if self.current_image_path:
            stem = Path(self.current_image_path).stem
            default_name = f"{stem}_result.jpg"
        
        filepath = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=filetypes
        )
        
        if filepath:
            try:
                cv2.imwrite(filepath, self.detection_result['result_image'])
                self.status_var.set(f"结果已保存: {filepath}")
                messagebox.showinfo("成功", "检测结果已保存！")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")


def main():
    """主函数"""
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"警告: 模型文件不存在: {MODEL_PATH}")
        print("请先运行训练脚本")
    
    # 创建窗口
    root = tk.Tk()
    
    # 设置 DPI 感知（Windows）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # 创建应用
    app = RoadDamageDetectionApp(root)
    
    # 运行
    print("\n启动桌面应用...")
    print("请使用界面上的按钮选择图片进行检测\n")
    
    root.mainloop()


if __name__ == '__main__':
    main()
