# Mix_cre: An Interpretable Approach for Accurate Prediction of Plant Regulatory Sequences with Cross-Species Generalization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)


Mix_cre 是一个面向植物顺式调控序列（cis-regulatory sequences / CREs）的轻量级深度学习框架：  
- 用 **Transformer** 捕捉局部 motif 及其组合规律  
- 用 **选择性状态空间模型（SSM / Mamba）** 以线性复杂度整合 kb 级长程依赖  
- 通过 **多任务学习（表达回归 + 物种分类）** 提升跨物种泛化与表征可复用性  
- 提供 **网页端离线部署** 与 **三类互补可解释分析（CRE scanning / Occlusion / ISM）**，支撑“预测—定位—改造”的连续流程

---

## 🌟 核心特性

- **混合架构**：Transformer + SSM(Mamba) 兼顾局部语法与长程依赖
- **多任务学习**：表达量回归（MSE）+ 物种分类（Cross Entropy），学习共享且物种可分的表征
- **跨物种泛化**：在拟南芥 / 水稻 / 番茄 / 玉米四物种联合数据上统一训练与评估
- **高参数效率**：在论文对比中，参数量仅为大型预训练模型 AgroNT 的约 2%（以文档表述为准）
- **可解释性与落地**：集成三类解释工具并提供 Web UI，支持导出 CSV/PNG 结果，便于统计、可视化与后续实验设计

---

## 🧬 任务与数据设定（论文协议）

### 主任务：跨物种表达量预测 + 物种识别（多任务）

- **输入**：以 TSS 为中心的调控序列，总长度约 3 kb  
  - 上游约 2000 bp + 下游约 1000 bp（文档描述）
- **回归标签**：多组织/多条件表达量取最大值，并做 `log2` 变换与标准化（文档描述）
- **数据划分**：每个物种约 80% / 10% / 10% 的 train/val/test 随机划分（保证比例一致）
- **评估指标（回归）**：Spearman ρ / Pearson r / MSE  
- **评估指标（分类）**：Accuracy / Precision / Recall / F1 / AUC（宏平均）

### 下游迁移：可用与启动子强度预测（可选）

- 输入序列更短，用于检验表征迁移与局部语法学习能力

---

## 📈 结果亮点（来自论文/文档汇总口径）



- **跨物种独立测试（主任务，宏平均）**：Spearman / Pearson 达到 **0.836 / 0.882**
- **物种分类总体表现**：Accuracy 约 **0.9870**
- **分物种分类正确率（混淆矩阵对角线）**：  
  - *A. thaliana* 98.9%  
  - *O. sativa* 95.5%  
  - *S. lycopersicum* 97.2%  
  - *Z. mays* 99.5%

---
### 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 1024 | 隐藏维度 |
| n_layers | 8 | Transformer 层数 |
| n_heads | 8 | 注意力头数 |
| tail_layers | 3 | SSM 层数 |
| vocab_size | 5 | 词表大小 (A, C, G, T, N) |
| max_len | 3001 | 最大序列长度 |
| parameters | ~22M | 总参数量 |

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/your-username/Mix_cre.git
cd Mix_cre
pip install -r requirements.txt
```

### 数据准备

准备包含以下列的 CSV 文件：
- `sequence`: DNA 序列（ATCGN 字符）
- `species`: 物种名称（如 "Arabidopsis_thaliana"）
- `tpm` 或 `tpm_max`: 表达值（TPM）

### 训练模型

```bash
python train.py --csv_path your_data.csv --out_dir checkpoints --epochs 50
```

### 推理预测

```python
from inference import load_model, predict_sequence

```

### Web 应用

```bash
# 启动 Web 服务器
python webapp/run.py

# Windows 用户
cd webapp && run.bat
```

访问 http://localhost:8000 使用网页界面。


## 🌱 支持物种

1. *拟南芥* (Arabidopsis thaliana)
2. *水稻* (Oryza sativa)
3. *番茄* (Solanum lycopersicum)
4. *玉米* (Zea mays)




## 🌐 Web 界面

Web 应用提供完整的植物基因组 CRE 分析平台，集成了先进的深度学习模型和多种解释性分析工具：

### 🎯 核心预测功能
- **🧬 启动子活性预测**：基于 Transformer+SSM 混合架构的实时预测
  - 支持 3001bp 序列输入（推荐：TSS 上游 2000bp + TSS + 下游 1000bp）
  - Z-score 标准化输出，直观的活性等级分类
  - 高精度预测：Spearman 相关系数 0.83，分类准确率 98.7%

- **🌱 多物种识别**：自动识别植物物种并提供物种特异性分析
  - 支持拟南芥、水稻、番茄、玉米四种主要作物
  - 物种分类概率输出，置信度评估
  - 物种特异性 TPM 估算和表达量转换

- **🔄 反向互补池化**：利用 DNA 双链特性增强预测稳定性
  - 正向和反向互补序列的平均预测
  - 提高预测鲁棒性和生物学合理性
  - 可选择开启/关闭该功能

### 🔬 模型解释性分析工具

#### 🔍 Occlusion 分析（遮挡分析）
滑动窗口遮挡法，通过系统性遮挡序列片段来量化每个区域对预测结果的贡献：
- **工作原理**：逐步遮挡序列片段，观察预测值变化
- **参数设置**：
  - 窗口大小：5-50bp（默认 15bp）
  - 步长：1-25bp（默认 15bp）
  - 批处理大小：16-128（默认 64）
- **结果展示**：
  - 交互式重要性热图可视化
  - 序列位置与重要性得分对应关系
  - 重要区域的颜色编码标注
- **应用场景**：识别关键调控区域，验证已知转录因子结合位点

#### 🧬 ISM 分析（饱和突变扫描）
In Silico Mutagenesis，系统性突变每个碱基位点，全面评估序列变异的功能影响：
- **工作原理**：将每个位点依次突变为其他三种碱基，计算预测值变化
- **技术特点**：
  - 高效批处理：支持大序列的快速扫描
  - 突变效应矩阵：4×L 维度的完整突变图谱
  - 智能优化：自适应批处理大小，避免内存溢出
- **结果分析**：
  - 突变效应热图：直观显示每个位点每种突变的影响
  - 重要性得分：基于突变敏感性的位点重要性排序
  - 功能位点识别：高敏感性位点可能为关键调控元件
- **生物学意义**：预测 SNP 功能影响，指导基因编辑设计

#### ⚡ Gradient×Input 分析（梯度重要性分析）
基于深度学习梯度的快速重要性分析方法：
- **计算原理**：梯度 × 输入值，反映每个位点对预测结果的直接贡献
- **技术优势**：
  - 计算速度快：单次前向和反向传播即可完成
  - 内存效率高：适合长序列和大规模分析
  - 生物学解释性强：直接反映模型注意力分布
- **结果特点**：
  - 位点级重要性得分
  - 正负贡献区分：正值促进表达，负值抑制表达
  - 与实验验证的转录因子结合位点高度一致
- **应用价值**：快速筛选候选 CRE，为后续实验提供指导

#### 🎯 CRE 扫描（顺式调控元件识别）
基于 Gradient×Input 的智能峰值检测算法，自动识别变长 CRE 区域：
- **算法特点**：
  - 自适应阈值：基于重要性分布的动态阈值确定
  - 智能峰值检测：识别连续高重要性区域
  - 变长区域识别：不限制 CRE 长度，更符合生物学实际
- **参数配置**：
  - 扫描窗口：5-50bp（默认 21bp）
  - 扫描步长：1-25bp（默认 21bp）
  - 结果数量：10-200 个（默认 100 个）
  - 最小 CRE 长度：5bp
  - 最大间隔：3bp（合并相近区域）
- **结果输出**：
  - CRE 区域坐标（起始、结束位置）
  - 重要性统计（平均值、最大值、峰值位置）
  - 按重要性排序的候选 CRE 列表
  - 可视化展示：重要性曲线 + CRE 区域标注
- **生物学应用**：
  - 新 CRE 发现：识别未知调控元件
  - 功能验证指导：为实验设计提供精确坐标
  - 进化分析：比较不同物种间 CRE 保守性

### 🎨 用户界面特色

#### 现代化设计
- **植物生物学主题**：清新绿色渐变配色方案
- **响应式布局**：完美适配桌面、平板、手机设备
- **无障碍设计**：支持键盘导航和屏幕阅读器
- **暗色主题**：护眼的深色模式选择

#### 交互体验
- **实时预测**：无需页面刷新的即时结果展示
- **进度指示**：详细的分析进度和状态反馈
- **加载动画**：优雅的等待界面和过渡效果
- **错误处理**：友好的错误提示和恢复建议

#### 数据可视化
- **交互式图表**：基于 Chart.js 的高质量可视化
- **序列注释**：碱基级别的颜色编码和悬停提示
- **热图展示**：直观的重要性和突变效应可视化
- **缩放平移**：支持图表的交互式探索

#### 数据导出
- **多格式支持**：TSV、CSV、FASTA、PNG 格式导出
- **结果保存**：完整的分析结果和参数记录
- **图表导出**：高分辨率图表图像下载
- **批量处理**：支持多序列的批量分析和导出

### 🔧 高级功能

#### 参数优化
- **自适应批处理**：根据序列长度和系统资源自动调整
- **内存管理**：智能内存使用，避免大序列分析时的内存溢出
- **并行计算**：充分利用多核 CPU 和 GPU 加速

#### 质量控制
- **序列验证**：自动检测和清理无效字符
- **长度检查**：序列长度合理性验证和警告
- **结果置信度**：提供预测置信度和不确定性估计

#### 用户指导
- **使用提示**：详细的功能说明和参数建议
- **示例数据**：内置示例序列用于功能演示
- **帮助文档**：完整的使用指南和常见问题解答


## 📁 项目结构

```
Mix_cre/
├── README.md              # 本文件
├── requirements.txt       # Python 依赖
├── LICENSE               # MIT 许可证
├── model.py              # 模型架构
├── dataset.py            # 数据处理
├── losses.py             # 损失函数
├── metrics.py            # 评估指标
├── train.py              # 训练脚本
├── inference.py          # 推理工具
├── example.py            # 使用示例
├── test_model.py         # 模型测试
├── checkpoints/          # 模型权重
│   └── best.pt
└── webapp/               # Web 应用
    ├── api.py            # FastAPI 后端
    ├── run.py            # 服务器启动器
    ├── static/           # 前端文件
    └── requirements.txt
```

## 🛠️ 开发

### 测试

```bash
# 运行模型测试
python test_model.py

# 测试 Web API
python -m pytest webapp/tests/
```

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| MODEL_CKPT | checkpoints/best.pt | 模型检查点路径 |
| SEQ_LEN | 3001 | 输入序列长度 |
| PORT | 8000 | Web 服务器端口 |

## 📚 API 文档


#### 解释性分析
- `POST /api/explain/occlusion` - Occlusion 分析
- `POST /api/explain/ism` - ISM 饱和突变扫描
- `POST /api/cre/scan_gi` - CRE 顺式调控元件扫描

### 解释性分析接口

#### Occlusion 分析
```bash
curl -X POST http://localhost:8000/api/explain/occlusion \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": "ATCGATCG...",
    "window": 15,
    "stride": 15,
    "batch_size": 64
  }'
```

#### ISM 饱和突变扫描
```bash
curl -X POST http://localhost:8000/api/explain/ism \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": "ATCGATCG...",
    "batch_size": 128,
    "rc_pool": true
  }'
```

#### Gradient×Input 分析
```bash
curl -X POST http://localhost:8000/api/explain/gi \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": "ATCGATCG...",
    "rc_pool": true
  }'
```

#### CRE 扫描
```bash
curl -X POST http://localhost:8000/api/cre/scan_gi \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": "ATCGATCG...",
    "window": 21,
    "stride": 21,
    "top_k": 100,
    "rc_pool": true
  }'
```



## 🙏 致谢

- 受 Evo2 模型架构启发
- Mamba（状态空间模型）实现
- 感谢所有贡献者和开源社区

## 📞 联系方式

如有问题或建议，请在 GitHub 上提交 issue 或联系我们：[liuhuan01@139.com]

---

**用 ❤️ 为植物基因组学研究制作**