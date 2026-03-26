# 部署说明

## Render 免费部署步骤

### 1. 上传代码到 GitHub

```bash
cd /d D:\road_damage_detection

git init
git add .
git commit -m "Initial commit"

# 替换为你的 GitHub 仓库地址
git remote add origin https://github.com/你的用户名/road-damage-detector.git
git push -u origin main
```

### 2. 在 Render 上部署

1. 打开 https://render.com
2. 用 GitHub 账号登录
3. 点击 "New +" → "Web Service"
4. 选择你的仓库
5. 配置：
   - Name: road-damage-detector
   - Environment: Python
   - Build Command: `pip install -r requirements_render.txt`
   - Start Command: `python web_app.py`
6. 点击 "Create Web Service"

### 3. 等待部署完成

约 3-5 分钟后，会得到一个免费链接，例如：
`https://road-damage-detector.onrender.com`

---

## 本地运行

```bash
pip install -r requirements.txt
python web_app.py
```

访问 http://localhost:5000