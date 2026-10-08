# 数学公式符号识别 CNN 神经网络 — 工程化方案

> 作者视角：机器学习工程师
> 目标：识别数学公式符号。交付：可落地的整体架构、代码目录、最终超参数、训练方案。
> 说明：本文所有公式与配置均为可直接复制使用的工程规格，不含 LaTeX 数学排版。

---

## 0. 摘要与结论（先看这里）

你给的两份材料对应的是**两个不同粒度的任务**：

| 任务 | 输入 | 输出 | 模型 | 难度 |
|---|---|---|---|---|
| **A. 单符号分类** | 单符号裁剪图 64×64 | 一个符号类别标签 | 纯 CNN 分类器 | 低（首个里程碑） |
| **B. 整公式识别** | 整条公式图 64×256 | 完整 LaTeX 序列 | CNN 编码器 + Transformer 解码器 | 中高（你架构图所指） |

你给的架构图（ResNet-18 编码器 + 4 层 Transformer 解码器，自回归生成 LaTeX）**就是任务 B**。它是学界标准做法（WAP / Image-to-LaTeX 一脉，Deng et al. 2016 起）。

**推荐路线**：先做 A（1 周可跑通、可验收），再做 B（A 的 CNN 编码器直接复用为 B 的 backbone）。本文以 B 为主、A 为基线，两个都给出最终超参数。

---

## 1. 任务建模与需求澄清

### 1.1 任务 A：单符号分类（你说的"识别符号"的字面意思）

- 输入：单张灰度图 64×64（已裁剪到单个符号）。
- 输出：类别标签（如 `0`~`9`、`alpha`、`beta`、`+`、`-`、`\times`、`\int`、`\sum`、`\sqrt` 等）。
- 类别数：CROHME 标准 101 类；自定义通常 100~200 类。
- 本质：标准图像分类，纯 CNN。

### 1.2 任务 B：整公式识别（你架构图所指）

- 输入：整条公式灰度图，固定高 64、可变宽 ≤ 256。
- 输出：LaTeX 序列，自回归逐 token 生成。
- 关键难点：**结构性符号**（`\frac{}{}`、`\sqrt{}`、上下标 `_` `^`、`\sum_{}^{}`、`\int_{}^{}`、`\left(\right)`）不是"单个孤立符号"，必须靠序列模型还原结构。
- 本质：sequence-to-sequence（图像 → 序列）。

### 1.3 验收口径（决定一切）

| 任务 | 核心指标 | 达标线 |
|---|---|---|
| A | Top-1 Accuracy / macro-F1 | Acc ≥ 95%，F1 ≥ 0.9 |
| B | Exact Match（整条公式全对） | EM ≥ 60%（im2latex-100k 上） |
| B | Edit Distance（Levenshtein，归一化） | 平均 ≥ 0.85 |
| B | Image BLEU（把预测 LaTeX 渲染回图再算 BLEU） | ≥ 0.75 |

**工程铁律**：训练指标（loss、perplexity）只用于监控，**验收只看上表**。EM 太严，B 任务要用 Edit Distance + Image BLEU 作为主指标。

---

## 2. 数据方案

### 2.1 数据集选型

| 数据集 | 类型 | 规模 | 用途 |
|---|---|---|---|
| **CROHME** | 手写公式（含符号级标注） | ~10 万表达式 | A 的类别标签、B 的手写场景 |
| **im2latex-100k** | 印刷体公式 + LaTeX | 10 万对 | B 的主训练/测试集 |
| **自造合成数据** | LaTeX 渲染 + 增广 | 任意 | 补类别长尾、扩规模 |

**建议**：
- 任务 A：用 CROHME 的符号级标注（它自带 101 类符号框），或自己裁剪 im2latex 渲染结果。
- 任务 B：直接用 im2latex-100k（`train/val/test` 官方划分，已含 image 与 latex 两列）。

### 2.2 图像预处理流水线（统一）

1. **灰度化**：RGB → 单通道（数学符号无颜色信息）。
2. **归一化**：像素 / 255 到 [0,1]，再按 ImageNet 均值方差标准化（单通道均值 0.5、方差 0.5 即可）。
3. **尺寸规范化**：等比缩放至高 64，宽按比例；宽 > 256 时等比缩到 256；宽 < 256 时右侧补 0（左对齐，避免居中造成结构歧义）。
4. **增广（训练时）**：随机仿射（旋转 ±5°、缩放 0.9~1.1、平移 ±2%）、轻微弹性形变、随机亮度/对比度（±10%）、随机椒盐噪声（p=0.005）。
5. **输出**：`[1, 64, 256]` 的 float32 tensor（对齐你架构图的 `[B, 1, 64, 256]`）。

### 2.3 Tokenizer 设计（任务 B 关键）

LaTeX 不能按字符切，必须**先 tokenize**：

- 单字符 token：字母、数字、`+ - = ( ) [ ] { } _ ^ , . : ; /`。
- 多字符命令：`\frac \sqrt \sum \int \prod \lim \sin \cos \log \alpha \beta \gamma \times \cdot \div \leq \geq \neq \approx \infty \pm \rightarrow \left \right \mathrm \text` 等（从训练集收集，按频率排序，取前 N）。
- 保留 token：`<sos>`、`<eos>`、`<pad>`、`<unk>`。
- 词表规模 V ≈ 500（im2latex-100k 去低频后约 500）。
- 最大序列长度 L = 256，超长截断。

**Tokenizer 产物**：`tokenizer.json`（token→id、id→token、special tokens），训练/推理共用，保证一致。

---

## 3. 模型设计

### 3.1 任务 B：CNN 编码器 + Transformer 解码器（主模型）

```
输入 [B, 1, 64, 256]
   │  ResNet-18 简化版（去掉最后的全连接与全局池化）
   │  stride 依次 2,2,2 → 总 stride 8
   ▼
特征图 [B, 512, 8, 32]
   │  展平（按空间顺序）→ [B, 256, 512]（N = 8×32 = 256 个位置）
   │  线性投影 512 → d_model=256  → [B, 256, 256]
   │  加 2D 正弦位置编码（行编码 + 列编码拼接/相加）
   ▼
Transformer 解码器（4 层，8 头，自回归）
   │  <sos> + 已生成 token → cross-attention 到编码特征
   ▼
每步输出 [B, V] 的 logits → 训练用 teacher forcing，推理 greedy/beam
```

**关键工程细节**：

- **编码器 output channel**：ResNet-18 最后一层 512 通道；若换 ResNet-34 也是 512。用 512→256 的 1×1 卷积或线性层降维到 d_model。
- **2D 位置编码**：不要用 1D 编码。用标准正弦位置编码分别算行(8)与列(32)，把两者拼接后线性映射，或直接相加。**手写公式里"上下标在竖直方向的位置"是强信号**，2D 编码不能省。
- **解码器**：标准 Pre-LN Transformer decoder，`d_model=256, nhead=8, nlayer=4, d_ff=1024`。加 `dropout=0.2`。
- **teacher forcing 比例**：训练 100% teacher forcing，加 `label smoothing=0.1` 防过拟合与"曝光偏差"。
- **损失**：交叉熵，忽略 `<pad>` 位置（`ignore_index=pad_id`）。

### 3.2 任务 A：单符号分类 CNN（基线 / 编码器复用）

```
输入 [B, 1, 64, 64]
   │ Conv(1→32,3,pad1) BN ReLU  MaxPool2 → 32×32
   │ Conv(32→64,3,pad1) BN ReLU  MaxPool2 → 16×16
   │ Conv(64→128,3,pad1) BN ReLU  MaxPool2 → 8×8
   │ Conv(128→256,3,pad1) BN ReLU  MaxPool2 → 4×4
   │ GlobalAvgPool → 256 → Dropout(0.3) → FC(256→N_classes)
   ▼
logits [B, N_classes] → CrossEntropy
```

这个 CNN 的前 3 个 block 与 B 的 ResNet 骨干同源，可作为 B 的**预训练起点**（在 CROHME 符号上预训练 backbone，再迁移到 B）。

---

## 4. 工程目录结构（在你建议上补齐）

```
math_formula_project/
├── configs/
│   ├── symbol_cls.yaml          # 任务 A 配置
│   └── im2latex.yaml            # 任务 B 配置
├── data/
│   ├── raw/                     # 原始数据（im2latex, crohme）
│   ├── processed/               # 预处理后的 tensor/h5 缓存
│   ├── images/                  # 规范化后的图
│   └── labels.json
├── models/
│   ├── cnn_classifier.py        # 任务 A：纯 CNN
│   ├── encoder.py               # 任务 B：ResNet 编码器 + 2D 位置编码
│   ├── decoder.py               # 任务 B：Transformer 解码器
│   └── seq2seq.py               # 组装 encoder+decoder 的 forward
├── utils/
│   ├── tokenizer.py             # LaTeX tokenizer + 读写 tokenizer.json
│   ├── dataset.py               # Dataset/DataLoader + 增广
│   ├── metrics.py               # EM / EditDistance / ImageBLEU / Acc / F1
│   ├── augmentation.py          # 仿射、弹性形变
│   └── logger.py                # tensorboard + wandb 封装
├── train_classifier.py          # 任务 A 训练入口
├── train_seq2seq.py             # 任务 B 训练入口
├── inference.py                 # 加载 ckpt，greedy/beam 推理，导出 ONNX
├── eval.py                      # 离线评估，输出 metrics 报告
├── requirements.txt             # torch, torchvision, transformers, ...
├── README.md
└── report/                      # 训练曲线、混淆矩阵、bad case 图
```

**工程规范**：
1. **配置驱动**：所有超参数进 YAML，代码里用 `@dataclass` 解析，禁止硬编码。
2. **可复现**：`seed=42`，`torch.use_deterministic_algorithms(True)`（必要时关闭以换速度）。
3. **断点续训**：每 epoch 存 `last.ckpt`，每个指标最优存 `best_EM.ckpt`，记录 `global_step` 与 optimizer 状态。
4. **缓存预处理**：图片预处理费时，首轮写入 `.h5`/`lmdb`，后续直接读缓存，训练提速 3~5 倍。

---

## 5. 训练方案

### 5.1 硬件与规模假设

- 单卡 A100 40G 或 2×3090 24G。
- 任务 B 单卡 batch 64；不够用 `gradient_accumulation` 凑等效 batch。
- 训练时长：任务 B ~50 epoch，单卡约 12~20 小时。

### 5.2 训练循环关键点

1. **混合精度**：`torch.cuda.amp`（bf16），loss 用 `GradScaler`。
2. **梯度裁剪**：`clip_grad_norm_ = 1.0`。
3. **学习率调度**：warmup 2000 step → cosine 衰减到 1e-6（见 §6）。
4. **early stopping**：val Edit Distance 连续 10 epoch 不提升则停。
5. **验证频率**：每 1 epoch 跑一次 val；每 5 epoch 存一次 bad case 可视化。
6. **数据并行**：多卡用 `DistributedDataParallel`（单机多卡即可），不用 `DataParallel`。

### 5.3 推理与部署

- **解码策略**：默认 greedy（快）；要求质量用 beam search（width=5，加长度惩罚 0.6）。
- **导出**：训练后 `torch.onnx.export` / `TorchScript`，供 C++ / 服务端部署。
- **服务化**：FastAPI 包一层，输入 base64 图，输出 LaTeX 字符串 + 置信度。
- **延迟预算**：单条公式（64×256）greedy 推理 < 20ms（A100），beam=5 < 50ms。

---

## 6. 最终超参数设置（可直接使用）

### 6.1 任务 B（整公式识别，主模型）

| 超参数 | 取值 | 说明 |
|---|---|---|
| input size | 1×64×256 | 灰度、高固定、宽补 0 |
| encoder backbone | ResNet-18（简化） | 去尾部 FC/池化，总 stride 8 |
| encoder output | 512 通道 × 8 × 32 | 展平成 N=256 位置 |
| 位置编码 | 2D 正弦（行+列） | 竖直方向信息关键 |
| d_model | 256 | 编码特征 512→256 投影后 |
| decoder 层数 | 4 | |
| 注意力头 | 8 | |
| d_ff | 1024 | |
| dropout | 0.2 | 编码/解码/注意力统一 |
| vocab size V | ~500 | 含 4 个 special token |
| max seq len L | 256 | |
| embedding | 256（与 d_model 同） | 共享权重可减参，不强制 |
| loss | CrossEntropy | `ignore_index=pad_id` |
| label smoothing | 0.1 | 防曝光偏差 |
| optimizer | AdamW | β=(0.9, 0.98) |
| lr | 1e-4 | 峰值 |
| lr schedule | warmup 2000 step + cosine → 1e-6 | |
| weight decay | 1e-4 | |
| grad clip | 1.0 | |
| batch size | 64 | 不够用 grad_accum 补 |
| epochs | 50 | early stopping 10 |
| precision | bf16 AMP | |
| seed | 42 | |
| beam width（推理） | 5 | length penalty 0.6 |

### 6.2 任务 A（单符号分类，基线）

| 超参数 | 取值 |
|---|---|
| input size | 1×64×64 |
| 通道 | 32→64→128→256 |
| 全连接 | 256 → N_classes（N≈101） |
| dropout | 0.3 |
| loss | CrossEntropy + label smoothing 0.1 |
| optimizer | AdamW lr=1e-3 |
| lr schedule | cosine → 1e-5 |
| batch size | 128 |
| epochs | 50 |
| 增广 | 旋转±10°、缩放0.8~1.2、弹性形变、噪声 |

---

## 7. 评估与验收

### 7.1 指标实现要点（utils/metrics.py）

- **Accuracy / F1**：任务 A，`sklearn.metrics`。
- **Exact Match**：预测 LaTeX 与真值逐字符全等。
- **Edit Distance**：Levenshtein，归一化 `1 - dist/max(len(pred),len(gt))`。
- **Image BLEU**：把预测 LaTeX 与真值 LaTeX 都用 `matplotlib`/KaTeX 渲染成图，算图像 BLEU（im2latex 官方口径，比字符串 BLEU 更贴合"看起来对不对"）。
- **混淆矩阵**：任务 A 必须出，定位易混对（如 `\times` vs `x`、`\leq` vs `<`、`0` vs `O`）。

### 7.2 达标标准（复述）

| 任务 | 主指标 | 达标 |
|---|---|---|
| A | Top-1 Acc / macro-F1 | ≥95% / ≥0.9 |
| B | EM / Edit Dist / Image BLEU | ≥60% / ≥0.85 / ≥0.75 |

### 7.3 交付物清单

- `report/train_curves.png`：loss / EM / Edit Distance 曲线。
- `report/confusion_matrix.png`（任务 A）。
- `report/bad_cases/`：50 张预测错误样例（图 + 真值 + 预测）。
- `best_EM.ckpt` + `tokenizer.json` + `onnx` 导出。

---

## 8. 风险与备选方案

| 风险 | 对策 |
|---|---|
| im2latex 印刷体太干净，泛化到拍照/手写差 | 加 CROHME 手写数据 + 强增广（弹性形变、噪声、模糊） |
| EM 长期上不去 | 主指标切到 Edit Distance + Image BLEU；加 beam search |
| 长公式（宽>256）被截断 | 放宽到 512，或用可伸缩编码器；先保证 ≤256 场景 |
| 结构性错误（`\frac` 的分子分母错位） | 数据里按结构 token 均衡采样；bad case 驱动增广 |
| 单卡显存不足 | batch 64→32 + grad_accum 2；bf16 已省一半显存 |
| 类别长尾（低频符号学不好） | 类别加权采样 / focal loss（任务 A） |

---

## 9. 里程碑计划（供排期）

| 阶段 | 内容 | 产出 | 时间 |
|---|---|---|---|
| M1 | 任务 A：数据裁剪 + CNN 分类器 | 可验收的分类模型 | 1 周 |
| M2 | 任务 B：tokenizer + dataset + seq2seq 跑通 | 能训练、能推理的骨架 | 1 周 |
| M3 | 任务 B：调参 + 增广 + 达标 | EM≥60% 的模型 | 2~3 周 |
| M4 | 部署：ONNX 导出 + FastAPI + 评测报告 | 可上线服务 | 1 周 |

---

## 附录 A：关键代码骨架

```python
# models/seq2seq.py  —— forward 主干（示意）
class Im2Latex(nn.Module):
    def __init__(self, enc, dec, d_model=256):
        super().__init__()
        self.enc = enc            # ResNet-18 -> [B,512,8,32]
        self.proj = nn.Conv2d(512, d_model, 1)   # -> [B,256,8,32]
        self.pos = Pos2D(h=8, w=32, d=d_model)   # 2D 正弦
        self.dec = dec            # TransformerDecoder 4层8头

    def encode(self, x):
        f = self.proj(self.enc(x))          # [B,d,8,32]
        f = f.flatten(2).transpose(1,2)     # [B,256,d]
        return self.pos(f)

    def forward(self, img, tgt, tgt_mask):
        mem = self.encode(img)
        return self.dec(tgt, mem, tgt_mask) # [B,L,V]
```

```python
# utils/metrics.py —— Edit Distance 归一化
import Levenshtein
def edit_dist_norm(pred, gt):
    return 1 - Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1)
```

```python
# train_seq2seq.py —— 训练主循环（示意）
for img, tgt in loader:
    img, tgt = img.cuda(), tgt.cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(img, tgt[:, :-1], tgt_mask)     # teacher forcing
        loss = F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                               ignore_index=pad_id)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); scheduler.step()
```

---

## 附录 B：一句话总结

- 你说的"识别符号"分两层：**单符号分类**（纯 CNN，先做）与**整公式转 LaTeX**（CNN+Transformer seq2seq，你架构图所指，后做）。
- 主模型超参数已锁定在 §6.1，直接复制即可跑通 im2latex-100k。
- 验收不看 loss，看 **EM / Edit Distance / Image BLEU**（任务 B）与 **Acc / F1**（任务 A）。
- 工程上把"配置化、可复现、断点续训、缓存预处理、ONNX 部署"当作与模型同等重要的一等公民。
