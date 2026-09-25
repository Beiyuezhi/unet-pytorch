# AB1 1D U-Net 自动裁剪

这是在原项目基础上新增的 **Sanger AB1 一维 U-Net baseline**。

## 目标

训练数据只要求成对的：

```text
raw_dir/
  sample001.ab1
  sample002.ab1

trimmed_dir/
  sample001.ab1
  sample002.ab1
```

其中：

- `raw_dir`：未裁剪 AB1
- `trimmed_dir`：人工已经裁剪好的同名 AB1
- 已裁剪 AB1 只用于自动生成训练标签
- 模型训练和正式推理时，输入特征只来自未裁剪 AB1

## 当前模型

第一版只使用标准 **1D U-Net**：

```text
未裁剪 AB1
   ↓
base-level 9通道特征
   ↓
1D U-Net
32 → 64 → 128 → 256 → 512
   ↓
每个 base 的 keep/discard logit
   ↓
取最长连续 keep 区间
   ↓
start / end
```

当前 9 个输入通道：

1. A base one-hot
2. C base one-hot
3. G base one-hot
4. T base one-hot
5. A 峰位信号强度
6. C 峰位信号强度
7. G 峰位信号强度
8. T 峰位信号强度
9. Phred quality

模型输出长度与原始 AB1 的 called-base 数一致。

## 标签如何自动生成

程序会把人工裁剪后的 AB1 base-call 序列与未裁剪 AB1 base-call 序列做 local alignment：

```text
raw:
NNNNNNACGTACGTACGTACGTNNNN

trimmed:
      ACGTACGTACGTACGT
```

然后生成：

```text
00000011111111111111110000
```

因此不需要额外人工标注峰异常、噪声、双峰等类别。

注意：这个 alignment 只用于 **训练标签生成**。正式推理时不需要 trimmed AB1。

## 安装

推荐新环境，不要直接依赖原仓库中较老的版本锁定：

```bash
pip install -r requirements_ab1.txt
```

## 训练

```bash
python train_ab1.py \
  --raw-dir data/raw \
  --trimmed-dir data/trimmed \
  --output-dir logs_ab1 \
  --epochs 100 \
  --batch-size 8
```

输出：

```text
logs_ab1/
  best.pth
  last.pth
  history.json
```

验证指标包括：

- interval IoU
- start MAE
- end MAE
- validation loss

## 推理

```bash
python predict_ab1.py \
  --ab1 data/raw/example.ab1 \
  --model logs_ab1/best.pth
```

输出示例：

```text
bases=1229
start=41
end=836
kept_bases=795
trimmed_sequence=...
```

## 目前第一版明确没有加入

为了先得到一个干净 baseline，目前没有加入：

- Residual block
- Dilated bottleneck
- TCN / MS-TCN++
- Transformer
- Boundary auxiliary head
- 双分支 encoder

后续应先用真实 train/val/test 数据测出本版结果，再逐项增加模块比较提升。

## 关键文件

```text
nets/unet_1d.py
utils/ab1_features.py
utils/ab1_alignment.py
datasets/ab1_dataset.py
train_ab1.py
predict_ab1.py
requirements_ab1.txt
```


## 边界优化版训练

当前训练脚本默认对真实裁剪起点和终点附近进行额外 BCE 加权：

- `--boundary-radius 12`：start/end 两侧各 12 bp
- `--boundary-weight 4.0`：边界区域 BCE 权重为普通位置的 4 倍
- Dice loss 仍然保留
- 默认 loss：`0.5 * boundary-weighted BCE + 0.5 * Dice`

支持 early stopping，但默认关闭（`--early-stopping-patience 0`），因此默认会完整跑满 `--epochs`。如需提前停止，可手动设置例如 `--early-stopping-patience 12`。

推荐 CPU 训练命令：

```powershell
python train_ab1.py \
  --raw-dir data/raw \
  --trimmed-dir data/trimmed \
  --output-dir logs_ab1_boundary \
  --epochs 100 \
  --batch-size 8 \
  --device cpu
```

新增验证指标：

- `boundary_mae`：start/end MAE 的平均
- `start_bias` / `end_bias`：有符号偏差；正数表示预测位置偏后，负数表示偏前
- `start_within_5bp/10bp/20bp`
- `end_within_5bp/10bp/20bp`

权重文件：

- `best.pth`：按最低 boundary MAE 保存
- `best_boundary.pth`：与 best.pth 相同，明确表示边界最佳
- `best_iou.pth`：按最高 interval IoU 保存
- `last.pth`：最后一个 epoch


## 1D ResNet50 U-Net 对比实验

已增加 `nets/unet_resnet_1d.py`，编码器采用标准 ResNet-50 bottleneck 深度：

```text
[3, 4, 6, 3]
```

所有 2D 操作都改为 1D，输入仍然是相同的 9 通道 AB1 base-level 特征，decoder 使用 U-Net skip connection。

训练 ResNet50：

```powershell
python train_ab1.py \
  --raw-dir data/raw \
  --trimmed-dir data/trimmed \
  --output-dir logs_ab1_resnet50 \
  --epochs 100 \
  --batch-size 8 \
  --device cpu \
  --backbone resnet50
```

原始 baseline 仍可使用：

```powershell
python train_ab1.py \
  --raw-dir data/raw \
  --trimmed-dir data/trimmed \
  --output-dir logs_ab1_plain \
  --epochs 100 \
  --batch-size 8 \
  --device cpu \
  --backbone plain
```

checkpoint 会保存 `backbone` 字段，`predict_ab1.py` 会自动选择正确的网络结构。


## 1D ResNet101 U-Net 对比实验

ResNet101 已支持，encoder stage 深度为：

```text
[3, 4, 23, 3]
```

与 ResNet50 的 `[3, 4, 6, 3]` 相比，主要是第三个 stage 从 6 个 bottleneck 增加到 23 个。

训练命令：

```powershell
python train_ab1.py \
  --raw-dir data/raw \
  --trimmed-dir data/trimmed \
  --output-dir logs_ab1_resnet101 \
  --epochs 100 \
  --batch-size 8 \
  --device cpu \
  --backbone resnet101
```

如果 CPU 内存或速度压力较大，可把 batch size 改为 4 或 2。

当前可选 backbone：

```text
plain
resnet50
resnet101
```

checkpoint 会保存 backbone，`predict_ab1.py` 会自动恢复对应的 1D U-Net 结构。
