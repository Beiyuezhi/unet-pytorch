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
