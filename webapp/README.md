# HMamba

Run a minimal web service that loads your trained Transformer+SSM checkpoint and exposes:
- POST /api/predict: predict normalized expression (z-scored log1p TPM)
- POST /api/explain/gi: Gradient × Input importance explanation
- POST /api/explain/occlusion: occlusion-based importance (alternative)
- GET /api/health
- Static demo UI at root `/`

## 快速开始 (Quickstart)

### 方法1: 使用启动脚本（推荐）

**Windows:**
```bash
# 双击运行或在命令行执行
webapp\run.bat
```

**Linux/Mac:**
```bash
# 从项目根目录执行
python webapp/run.py
```

### 方法2: 手动启动

```bash
# 从项目根目录
cd webapp

# 安装依赖（如果还没安装）
pip install -r requirements.txt

# 设置环境变量（可选，脚本会自动查找）
export MODEL_CKPT="../results/best.ckpt"  # 或 results/checkpoints/best.ckpt
export SEQ_LEN=3000

# 启动API服务器
uvicorn webapp.api:app --host 0.0.0.0 --port 8000
```

### 访问Web界面

启动后，在浏览器中访问：
- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs (Swagger UI)
- **健康检查**: http://localhost:8000/api/health

## 环境变量配置

- `MODEL_CKPT`: 模型检查点路径（默认: `../results/best.ckpt` 或 `../results/checkpoints/best.ckpt`）
- `SEQ_LEN`: 序列长度（默认: 3000）
- `HOST`: 服务器地址（默认: 0.0.0.0）
- `PORT`: 服务器端口（默认: 8000）

## API测试

```bash
# 健康检查
curl http://localhost:8000/api/health

# 预测
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ACGTACGTACGT", "rc_pool": true}'

# GI解释
curl -X POST http://localhost:8000/api/explain/gi \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ACGTACGTACGT", "rc_pool": true}'

# 遮挡法解释（备选）
curl -X POST http://localhost:8000/api/explain/occlusion \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ACGTACGTACGT", "window": 21, "stride": 21, "batch_size": 64}'
```

## 注意事项

- API会自动从检查点推断超参数（num_species, d_model, n_layers, tail类型等）以匹配训练时的模型配置
- 如果训练时使用了不同的 `seq_len`，请设置 `SEQ_LEN` 环境变量
- 生产环境建议使用 Nginx 作为反向代理，将静态文件服务到 `/`，API路径代理到 `/api/*`
- 确保已安装 PyTorch 和项目依赖（evo2_mix 模块）




