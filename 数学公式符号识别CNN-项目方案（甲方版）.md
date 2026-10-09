# 数学公式符号识别系统 —— 项目技术方案

**适用范围**：数学公式符号识别 CNN 神经网络的设计、开发、训练与部署
方案参考了NeuroTeX的模型设计架构，其使用ResNet-18 CNN作为编码器，配合一个Transformer解码器，并且包含数据生成、训练、推理的完整代码，但是在CNN输出的图片特征转化为transform架构的过程中采用了裁剪转化的方式导致损失了一些关键的context，当其面对一些长文本的图片时会出现错误，在其基础上我们做了最关键的处理CNN卷积层和transform层的数据转化衔接。
同时由于我们算力资源比较紧张，为了保持一定的模型能力，我们查找了Mini-CoMER的结构，采用了CNN编码器+Transformer解码器的结构，还加入了一个注意力精炼模块（ARM） 来优化效果，非常适合在资源有限的环境下训练。我们设置了两个任务，任务A是做纯classfier，只做CNN卷积的softmax分类，不做seq2seq的输出，训练难度小，任务2是做全公式的手写体识别，输出latex码，训练难度，模型设计较大，前期我们会先做classfier，后面根据时间、成本来考虑是否推进任务B。

---

## 目录

1. [项目背景与建设目标](#一项目背景与建设目标)
2. [建设范围与任务边界](#二建设范围与任务边界)
3. [总体技术路线](#三总体技术路线)
4. [数据建设方案](#四数据建设方案)
5. [模型设计](#五模型设计)
6. [工程目录与代码规范](#六工程目录与代码规范)
7. [训练方案](#七训练方案)
8. [超参数配置](#八超参数配置)
9. [验收标准与交付物](#九验收标准与交付物)
10. [实施计划与里程碑](#十实施计划与里程碑)
11. [风险识别与应对](#十一风险识别与应对)
12. [附录：接口与代码骨架](#附录接口与代码骨架)

---

## 一、项目背景与建设目标

### 1.1 项目背景

本项目建设一套用于**数学公式符号识别**的深度学习系统，覆盖两类输入场景：

- 单个符号的裁剪图像，输出其类别标签；
- 整条数学公式图像，输出对应的 LaTeX 文本。

### 1.2 建设目标

1. 完成单符号分类模型，识别精度 Top-1 准确率 ≥ 95%；
2. 完成整公式识别模型，Exact Match ≥ 60%（在 im2latex-100k 基准上）；
3. 形成可复现、可维护、可部署的工程交付物。

---

## 二、建设范围与任务边界

本项目包含两个任务，粒度与验收口径不同：

| 任务 | 输入 | 输出 | 模型 | 定位 |
|---|---|---|---|---|
| A. 单符号分类 | 单符号裁剪图 64×64 | 符号类别标签 | 纯 CNN 分类器 | 首个里程碑 |
| B. 整公式识别 | 整公式图 64×256 | LaTeX 序列 | CNN 编码器 + Transformer 解码器 | 主体工程 |

> 说明：任务 A 的 CNN 骨干在任务 B 中复用为编码器，二者共享底层视觉特征。

### 2.1 验收口径

| 任务 | 主指标 | 达标线 |
|---|---|---|
| A | Top-1 准确率 / macro-F1 | ≥ 95% / ≥ 0.9 |
| B | Exact Match | ≥ 60% |
| B | 归一化 Edit Distance | ≥ 0.85 |
| B | Image BLEU | ≥ 0.75 |

> 原则：训练损失（loss、perplexity）仅用于过程监控，验收仅依据上表指标。整公式场景 Exact Match 过于严格，以 Edit Distance 与 Image BLEU 为主指标。

---

## 三、总体技术路线

### 3.1 项目思维导图

```
数学公式符号识别系统
│
├── 任务A：单符号分类
│   ├── 输入：64×64 单符号图
│   ├── 模型：纯 CNN 分类器
│   ├── 指标：Top-1 Acc / F1
│   └── 定位：首个里程碑，CNN 骨干复用
│
├── 任务B：整公式识别
│   ├── 输入：64×256 整公式图
│   ├── 模型：CNN 编码器 + Transformer 解码器
│   ├── 输出：LaTeX 序列
│   └── 指标：EM / Edit Distance / Image BLEU
│
├── 数据建设
│   ├── 数据集：CROHME / im2latex-100k / 合成
│   ├── 预处理：灰度、归一化、尺寸规范
│   └── 词表：LaTeX tokenizer
│
├── 工程设施
│   ├── 配置管理（YAML）
│   ├── 训练调度（AMP / DDP / 断点续训）
│   └── 模型仓库与部署（ONNX / FastAPI）
│
└── 交付与验收
    ├── 模型权重 + 词表 + 部署产物
    └── 评估报告 + 曲线 + 混淆矩阵
```

### 3.2 整体架构图（任务 B）

```
输入图片 [B, 1, 64, 256]
        │
        ▼
┌─────────────────────┐
│   CNN 编码器         │  ← 卷积提取视觉特征
│   (ResNet-18 简化版) │
└─────────────────────┘
        │
        ▼  特征图 [B, 512, 8, 32]
┌─────────────────────┐
│  展平为序列          │  → [B, 256, 512]
│  + 2D 位置编码       │  → [B, 256, 256]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Transformer 解码器  │  ← 自回归生成 LaTeX
│  (4 层，8 个注意力头)│
└─────────────────────┘
        │
        ▼
每步输出 token 概率 → 取 argmax / beam → 拼接为 LaTeX 字符串
```

### 3.3 技术选型

| 模块 | 选型 | 说明 |
|---|---|---|
| 深度学习框架 | PyTorch 2.x/tensflow 2.1 | 训练 + 导出 |
| 视觉骨干 | ResNet-18（简化） | 去尾部 FC/池化，总 stride 8 |
| 序列模型 | Transformer Decoder | 自回归，Pre-LN |
| 位置编码 | 2D 正弦 | 竖直方向信息为关键信号 |
| 混合精度 | bf16 / AMP | 显存减半 |
| 并行 | DistributedDataParallel | 单机多卡 |
| 部署 | ONNX / TorchScript + FastAPI | 服务化 |

---

## 四、数据建设方案

### 4.1 数据流水线

```
原始数据（CROHME / im2latex-100k / 合成渲染）
        │
        ▼
灰度化 → 归一化（/255 → 标准化）
        │
        ▼
尺寸规范化（等比缩放至高度 64，宽度上限 256，右侧补 0）
        │
        ▼
训练增广（仿射 / 弹性形变 / 亮度对比度 / 噪声）  ← 仅训练集
        │
        ▼
Tokenizer（LaTeX → token 序列）
        │
        ▼
预处理缓存（.h5 / lmdb）→ DataLoader → 训练
```

### 4.2 数据集选型

| 数据集 | 类型 | 规模 | 用途 |
|---|---|---|---|
| CROHME | 手写公式（含符号级标注） | 约 10 万表达式 | 任务 A 类别标签、任务 B 手写场景 |
| im2latex-100k | 印刷体公式 + LaTeX | 10 万对 | 任务 B 主训练/测试集 |
| 合成数据 | LaTeX 渲染 + 增广 | 按需 | 补类别长尾、扩规模 |

### 4.3 预处理规格

| 步骤 | 规格 |
|---|---|
| 灰度化 | RGB → 单通道（符号无颜色信息） |
| 归一化 | 像素 / 255 → [0,1]，再标准化（均值 0.5、方差 0.5） |
| 尺寸规范 | 等比缩至高 64；宽 > 256 等比缩至 256；宽 < 256 右侧补 0（左对齐） |
| 增广 | 旋转 ±5°、缩放 0.9~1.1、平移 ±2%、弹性形变、亮度/对比度 ±10%、椒盐噪声 p=0.005 |

### 4.4 Tokenizer 规格（任务 B）

LaTeX 不按字符切分，须先 tokenize：

- 单字符 token：字母、数字、`+ - = ( ) [ ] { } _ ^ , . : ; /`；
- 多字符命令：`\frac \sqrt \sum \int \prod \lim \sin \cos \log \alpha \beta \gamma \times \cdot \div \leq \geq \neq \approx \infty \pm \rightarrow \left \right` 等（按训练集频率收集，取前 N）；
- 保留 token：`<sos>`、`<eos>`、`<pad>`、`<unk>`；
- 词表规模 V ≈ 500；
- 最大序列长度 L = 256。

产物：`tokenizer.json`，训练与推理共用，保证映射一致。

---

## 五、模型设计

### 5.1 任务 A：单符号分类 CNN

```
输入 [B, 1, 64, 64]
   │ Conv(1→32,3,pad1) + BN + ReLU + MaxPool2 → 32×32
   │ Conv(32→64,3,pad1) + BN + ReLU + MaxPool2 → 16×16
   │ Conv(64→128,3,pad1) + BN + ReLU + MaxPool2 → 8×8
   │ Conv(128→256,3,pad1) + BN + ReLU + MaxPool2 → 4×4
   │ GlobalAvgPool → 256 → Dropout(0.3) → FC(256→N_classes)
   ▼
logits [B, N_classes] → CrossEntropy
```
##当做任务B时会减少最大池化，直接将CNN输出转化为sequence输入给接下来的transform层
### 5.2 任务 B：CNN 编码器 + Transformer 解码器

```
输入 [B, 1, 64, 256]
   │ ResNet-18 简化版（stride 依次 2,2,2 → 总 stride 8）
   ▼
特征图 [B, 512, 8, 32]
   │ 展平 → [B, 256, 512]
   │ 线性投影 512 → 256
   │ 加 2D 正弦位置编码（行 + 列）
   ▼
Transformer 解码器（4 层、8 头、d_model=256、d_ff=1024、自回归、具体层数会根据接下来的实际情况进行增减）
   ▼
每步输出 [B, V] logits → 训练 teacher forcing / 推理 greedy 或 beam
```

**关键规格**：

| 组件 | 规格 |
|---|---|
| 编码器输出 | 512 通道 × 8 × 32，展平成 N=256 位置 |
| 位置编码 | 2D 正弦（行 8、列 32 分别编码后拼接/相加） |
| d_model | 256（编码特征 512→256 投影） |
| 解码器 | 4 层、8 头、d_ff=1024、dropout=0.2 |
| teacher forcing | 训练 100%，配 label smoothing=0.1 |
| 损失 | 交叉熵，ignore_index=pad_id |

> 说明：手写公式的上下标在竖直方向的位置是关键特征，因此必须采用 2D 位置编码而非 1D。

---

## 六、工程目录与代码规范

### 6.1 目录结构

```
math_formula_project/
├── configs/
│   ├── symbol_cls.yaml          # 任务 A 配置
│   └── im2latex.yaml            # 任务 B 配置
├── data/
│   ├── raw/                     # 原始数据
│   ├── processed/               # 预处理缓存（h5/lmdb）
│   ├── images/                  # 规范化图像
│   └── labels.json
├── models/
│   ├── cnn_classifier.py        # 任务 A：纯 CNN
│   ├── encoder.py               # 任务 B：ResNet 编码器 + 2D 位置编码
│   ├── decoder.py               # 任务 B：Transformer 解码器
│   └── seq2seq.py               # 组装 encoder + decoder
├── utils/
│   ├── tokenizer.py             # LaTeX tokenizer + tokenizer.json 读写
│   ├── dataset.py               # Dataset / DataLoader + 增广
│   ├── metrics.py               # EM / EditDist / ImageBLEU / Acc / F1
│   ├── augmentation.py          # 仿射、弹性形变
│   └── logger.py                # tensorboard + wandb 封装
├── train_classifier.py          # 任务 A 训练入口
├── train_seq2seq.py             # 任务 B 训练入口
├── inference.py                 # greedy/beam 推理 + ONNX 导出
├── eval.py                      # 离线评估，输出指标报告
├── requirements.txt
├── README.md
└── report/                      # 训练曲线、混淆矩阵、bad case
```

### 6.2 代码规范

1. **配置驱动**：全部超参数入 YAML，代码以 dataclass 解析，不硬编码；
2. **可复现**：seed=42，关键步骤 deterministic；
3. **断点续训**：每 epoch 存 `last.ckpt`，指标最优存 `best_*.ckpt`，记录 `global_step` 与 optimizer 状态；
4. **预处理缓存**：图片预处理首轮写入 `.h5`/`lmdb`，后续直接读缓存，训练提速 3~5 倍。

---

## 七、训练方案

### 7.1 训练流水线

```
配置 YAML → 构建模型 → 加载数据（缓存）
                │
                ▼
        训练循环（AMP / 梯度裁剪 / 学习率调度）
                │
                ▼
        每 epoch 验证（EM / Edit Distance）
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
   保存 best.ckpt    early stopping（10 epoch 不提升）
        │
        ▼
导出 ONNX / TorchScript → FastAPI 服务
```

### 7.2 训练规格

| 项目 | 规格 |
|---|---|
| 硬件 | 单卡 A100 40G 或 2×3090 24G |
| 混合精度 | bf16 AMP（GradScaler） |
| 梯度裁剪 | clip_grad_norm = 1.0 |
| 学习率调度 | warmup 2000 step + cosine 衰减至 1e-6 |
| 早停 | val Edit Distance 连续 10 epoch 不提升即停 |
| 验证频率 | 每 epoch 一次；每 5 epoch 输出 bad case 可视化 |
| 并行 | DistributedDataParallel（单机多卡） |
| 数据并行 | batch 不足时以 gradient_accumulation 补齐等效 batch |

### 7.3 推理与部署

- 解码：默认 greedy；质量优先用 beam search（width=5，长度惩罚 0.6）；
- 导出：ONNX / TorchScript；
- 服务化：FastAPI，输入 base64 图像，输出 LaTeX 字符串与置信度；
- 延迟预算：单条公式（64×256）greedy < 20ms，beam=5 < 50ms（A100）。

---

## 八、超参数配置

### 8.1 任务 B（整公式识别，主模型）

| 超参数 | 取值 | 说明 |
|---|---|---|
| input size | 1×64×256 | 灰度、高固定、宽补 0 |
| encoder | ResNet-18（简化） | 总 stride 8 |
| encoder output | 512 × 8 × 32 | 展平成 N=256 |
| 位置编码 | 2D 正弦 | 行 + 列 |
| d_model | 256 | 编码特征 512→256 投影 |
| decoder 层数 | 4 | |
| 注意力头 | 8 | |
| d_ff | 1024 | |
| dropout | 0.2 | 编码/解码/注意力统一 |
| vocab size V | ~500 | 含 4 个 special token |
| max seq len L | 256 | |
| loss | CrossEntropy | ignore_index=pad_id |
| label smoothing | 0.1 | |
| optimizer | AdamW | β=(0.9, 0.98) |
| lr | 1e-4 | 峰值 |
| lr schedule | warmup 2000 + cosine → 1e-6 | |
| weight decay | 1e-4 | |
| grad clip | 1.0 | |
| batch size | 64 | 不足用 grad_accum 补 |
| epochs | 50 | early stopping 10 |
| precision | bf16 AMP | |
| seed | 42 | |
| beam width（推理） | 5 | length penalty 0.6 |

### 8.2 任务 A（单符号分类，基线）

| 超参数 | 取值 |
|---|---|
| input size | 1×64×64 |
| 通道 | 32→64→128→256 |
| 全连接 | 256 → N_classes（N≈101） |
| dropout | 0.3 |
| loss | CrossEntropy + label smoothing 0.1 |
| optimizer | AdamW，lr=1e-3 |
| lr schedule | cosine → 1e-5 |
| batch size | 128 |
| epochs | 50 |
| 增广 | 旋转 ±10°、缩放 0.8~1.2、弹性形变、噪声 |

---

## 九、验收标准与交付物

### 9.1 指标实现

| 指标 | 说明 |
|---|---|
| Accuracy / F1 | 任务 A，sklearn.metrics |
| Exact Match | 预测 LaTeX 与真值逐字符全等 |
| Edit Distance | Levenshtein，归一化 1 − dist/max(len(pred), len(gt)) |
| Image BLEU | 预测与真值 LaTeX 渲染成图后计算图像 BLEU |
| 混淆矩阵 | 任务 A，定位易混对（如 ×/x、≤/<、0/O） |

### 9.2 交付物清单

- `report/train_curves.png`：loss / EM / Edit Distance 曲线；
- `report/confusion_matrix.png`（任务 A）；
- `report/bad_cases/`：50 张错误样例（图 + 真值 + 预测）；
- `best_EM.ckpt` + `tokenizer.json` + ONNX 导出产物；
- 评估报告（含达标结论）。

---

## 十、实施计划与里程碑

| 阶段 | 内容 | 产出 | 周期 |
|---|---|---|---|
| M1 | 任务 A：数据裁剪 + CNN 分类器 | 可验收的分类模型 | 1 周 |
| M2 | 任务 B：tokenizer + dataset + seq2seq 跑通 | 可训练、可推理骨架 | 1 周 |
| M3 | 任务 B：调参 + 增广 + 达标 | EM ≥ 60% 模型 | 2~3 周 |
| M4 | 部署：ONNX 导出 + FastAPI + 评测报告 | 可上线服务 | 1 周 |

```
实施排期（示意）
M1 ████
M2     ████
M3          ████████████
M4                       ████
```

---

## 十一、风险识别与应对

| 风险 | 应对措施 |
|---|---|
| 印刷体数据泛化到拍照/手写差 | 加入 CROHME 手写数据 + 强增广（弹性形变、噪声、模糊） |
| Exact Match 长期不达标 | 主指标切到 Edit Distance + Image BLEU，加 beam search |
| 长公式（宽>256）截断 | 放宽宽度至 512 或换可伸缩编码器，先保证 ≤256 场景 |
| 结构错误（frac 分子分母错位） | 按结构 token 均衡采样 + bad case 驱动增广 |
| 单卡显存不足 | batch 64→32 + grad_accum 2；bf16 减半显存 |
| 类别长尾（低频符号） | 类别加权采样 / focal loss（任务 A） |
主要会通过一些自己标注和噪音数据集来进行训练，增强模型抗干扰的能力

---

## 附录：接口与代码骨架

```python
# models/seq2seq.py —— 任务 B 主干（示意）
class Im2Latex(nn.Module):
    def __init__(self, enc, dec, d_model=256):
        super().__init__()
        self.enc = enc                 # ResNet-18 -> [B,512,8,32]
        self.proj = nn.Conv2d(512, d_model, 1)   # -> [B,256,8,32]
        self.pos = Pos2D(h=8, w=32, d=d_model)   # 2D 正弦位置编码
        self.dec = dec                 # TransformerDecoder 4 层 8 头

    def encode(self, x):
        f = self.proj(self.enc(x))           # [B,d,8,32]
        f = f.flatten(2).transpose(1, 2)     # [B,256,d]
        return self.pos(f)

    def forward(self, img, tgt, tgt_mask):
        mem = self.encode(img)
        return self.dec(tgt, mem, tgt_mask)  # [B,L,V]
```

```python
# utils/metrics.py —— 归一化 Edit Distance
import Levenshtein
def edit_dist_norm(pred, gt):
    return 1 - Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1)
```

```python
# train_seq2seq.py —— 训练主循环（示意）
for img, tgt in loader:
    img, tgt = img.cuda(), tgt.cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(img, tgt[:, :-1], tgt_mask)   # teacher forcing
        loss = F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                               ignore_index=pad_id)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); scheduler.step()
```
