# 路面病害检测系统

基于 YOLO 深度学习框架的路面病害自动检测系统，支持训练、推理、Web 应用和桌面应用。

## 项目特点

- 完整的数据集配置和训练流程
- 支持单张图片和批量图片检测
- 提供 Web 界面和桌面应用两种使用方式
- 支持模型导出（ONNX、TensorRT 等）
- 代码结构清晰，易于部署和维护

## 目录结构

```
road_damage_detection/
├── data.yaml                   # 数据集配置文件
├── train.py                    # 训练脚本
├── val.py                      # 验证脚本
├── test.py                     # 测试脚本
├── inference.py                # 批量推理脚本
├── inference_core.py           # 核心推理模块（Web和桌面共用）
├── web_app.py                  # Web 应用
├── desktop_app.py              # 桌面应用
├── export.py                   # 模型导出脚本
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
├── models/                     # 模型保存目录
│   └── road_damage_detection/  # 训练结果
├── outputs/                    # 输出结果目录
├── static/temp/                # Web 临时文件
└── templates/                  # Web 前端模板
```

## 环境配置

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 CUDA（GPU训练需要）

如果使用 NVIDIA GPU 训练，请安装对应版本的 CUDA 和 cuDNN：
- CUDA 11.8 或 12.1
- PyTorch CUDA 版本

## 快速开始

### 第一步：训练模型

```bash
python train.py
```

训练参数修改：编辑 `train.py` 文件顶部的配置参数

### 第二步：验证模型

```bash
python val.py
```

### 第三步：测试模型

```bash
python test.py
```

### 第四步：运行推理

**单张图片检测：**
```bash
python inference.py --mode single --input path/to/image.jpg
```

**批量文件夹检测：**
```bash
python inference.py --mode folder --input path/to/folder
```

## Web 应用使用

### 启动 Web 服务

```bash
python web_app.py
```

服务启动后，在浏览器中访问：
```
http://localhost:5000
```

### Web 功能

- 上传图片进行检测
- 显示原图和检测结果对比
- 展示检测类别和置信度
- 支持 API 接口调用

## 桌面应用使用

### 启动桌面应用

```bash
python desktop_app.py
```

### 桌面功能

- 选择本地图片进行检测
- 显示原图和检测结果
- 查看检测统计信息
- 导出检测结果图片

## 打包成 EXE

### 1. 安装打包工具

```bash
pip install pyinstaller auto-py-to-exe
```

### 2. 使用 GUI 工具打包

```bash
auto-py-to-exe
```

### 3. 或直接使用命令行打包

**打包桌面应用：**
```bash
pyinstaller --noconfirm --onefile --windowed --icon=icon.ico --add-data "models;models" --add-data "data.yaml;." desktop_app.py
```

**打包 Web 应用：**
```bash
pyinstaller --noconfirm --onefile --console --icon=icon.ico --add-data "models;models" --add-data "templates;templates" --add-data "static;static" --add-data "data.yaml;." web_app.py
```

## 模型导出

### 导出 ONNX 格式

```bash
python export.py
```

### 导出 TensorRT 格式

```bash
python export.py engine
```

### 批量导出所有格式

```bash
python export.py all
```

## 数据集说明

### 病害类别

| 类别 | 名称 | 说明 |
|------|------|------|
| D00 | 纵向裂缝 | 沿道路方向的裂缝 |
| D10 | 横向裂缝 | 垂直于道路方向的裂缝 |
| D20 | 网状裂缝 | 呈网状分布的裂缝 |
| D40 | 坑槽 | 路面凹陷破损 |
| D43 | 松散 | 路面材料松散 |
| D44 | 车辙 | 车轮碾压形成的凹槽 |
| D50 | 修补 | 路面修补区域 |

### 数据集配置

数据集配置文件 `data.yaml`：

```yaml
path: G:/ttt/yolo_dataset      # 数据集根目录
train: images/train            # 训练集
test: images/test              # 测试集
val: images/val                # 验证集

names:                         # 类别名称
  0: D00
  1: D10
  2: D20
  3: D40
  4: D43
  5: D44
  6: D50
nc: 7                          # 类别数量
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_TYPE | yolov8s.pt | 模型类型 |
| EPOCHS | 100 | 训练轮数 |
| BATCH_SIZE | 8 | 批次大小 |
| IMG_SIZE | 640 | 输入图像尺寸 |
| DEVICE | "0" | 训练设备 |
| OPTIMIZER | "AdamW" | 优化器 |

## 常见问题

### 1. CUDA out of memory

**解决方案：**
- 减小 BATCH_SIZE（建议：8GB显存用 8，16GB显存用 16）
- 减小 IMG_SIZE（尝试 480 或 320）
- 使用更小的模型（yolov8n.pt 代替 yolov8s.pt）

### 2. 模型加载失败

**解决方案：**
- 确认训练已完成
- 检查 `MODEL_PATH` 路径是否正确
- 确认模型文件存在

### 3. Web 应用无法访问

**解决方案：**
- 检查端口 5000 是否被占用
- 尝试修改 `web_app.py` 中的 PORT 参数
- 确保防火墙允许该端口

## 更新日志

### v1.0.0
- 初始版本发布
- 支持训练和推理
- 提供 Web 和桌面两种界面

## 技术栈

- **深度学习**：YOLOv8, PyTorch
- **后端**：Flask
- **前端**：HTML5, JavaScript, CSS
- **GUI**：tkinter
- **图像处理**：OpenCV, Pillow

## 许可协议

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎反馈。

---

**祝你使用愉快！**
