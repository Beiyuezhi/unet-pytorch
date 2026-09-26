# AB1 自动裁剪：增强版 1D ResNet U-Net

当前 AB1 主模型已经统一为增强版架构，支持：

- 1D ResNet50
- 1D ResNet101
- ECA channel attention
- Dilated Context Bottleneck（dilation 1/2/4/8）
- Attention U-Net skip gates
- segmentation head
- start/end boundary head

> 当前版本不兼容之前旧结构的 checkpoint，需要重新训练。

## 输入数据

训练数据目录：

```text
data/
  raw/
    sample001.ab1
    sample002.ab1
  trimmed/
    sample001.ab1
    sample002.ab1
```

`raw` 和 `trimmed` 中的 AB1 必须同名配对。

- raw：未裁剪 AB1
- trimmed：人工裁剪后的 AB1，只用于生成监督标签
- 推理时只需要 raw AB1

## 输入特征

每个 called base 使用 9 个特征通道：

1. A one-hot
2. C one-hot
3. G one-hot
4. T one-hot
5. A 峰位强度
6. C 峰位强度
7. G 峰位强度
8. T 峰位强度
9. Phred quality

## 当前网络结构

```text
                Raw AB1
                   ↓
             9-channel features
                   ↓
         1D ResNet50 / ResNet101
                   ↓
           ECA attention blocks
                   ↓
        Dilated Context Bottleneck
          dilation 1 / 2 / 4 / 8
                   ↓
        Attention U-Net Decoder
              ↑      ↑
        attention-gated skips
                   ↓
           shared feature map
              /          \
             /            \
 segmentation head     boundary head
      ↓                 ↓      ↓
   keep mask          start    end
```

ResNet stage depth：

```text
ResNet50  = [3, 4, 6, 3]
ResNet101 = [3, 4, 23, 3]
```

## Loss

总 loss：

```text
segmentation loss
+ 0.3 × normalized start boundary CE
+ 0.7 × normalized end boundary CE
```

其中 segmentation loss：

```text
0.5 × boundary-weighted BCE
+ 0.5 × Dice
```

默认 end head 权重大于 start head，因为当前业务中末端边界更难预测。

可通过：

```text
--start-head-weight
--end-head-weight
```

调整。

## ResNet50 训练

PowerShell：

```powershell
python train_ab1.py `
  --raw-dir data/raw `
  --trimmed-dir data/trimmed `
  --output-dir logs_ab1_resnet50_enhanced `
  --epochs 100 `
  --batch-size 8 `
  --device cpu `
  --backbone resnet50
```

## ResNet101 训练

```powershell
python train_ab1.py `
  --raw-dir data/raw `
  --trimmed-dir data/trimmed `
  --output-dir logs_ab1_resnet101_enhanced `
  --epochs 100 `
  --batch-size 8 `
  --device cpu `
  --backbone resnet101
```

CPU 下如果 ResNet101 太慢或内存压力较大，可改：

```text
--batch-size 4
```

或：

```text
--batch-size 2
```

## Early stopping

默认关闭：

```text
--early-stopping-patience 0
```

因此默认会完整跑满 `--epochs`。

需要时可手动开启：

```text
--early-stopping-patience 12
```

## 输出

训练目录中会生成：

```text
best.pth
best_boundary.pth
best_iou.pth
last.pth
history.json
```

其中：

- `best.pth`：最低 boundary MAE
- `best_boundary.pth`：同 best.pth
- `best_iou.pth`：最高 interval IoU
- `last.pth`：最后一个 epoch

## 主要指标

训练日志会显示：

- start_mae
- end_mae
- boundary_mae
- interval_iou
- end_bias
- end<=10bp
- end<=20bp
- boundary_fallback_rate

boundary head 是主边界预测器。

如果 boundary head 偶尔给出无效区间（end <= start），推理会自动回退到 segmentation mask 的最长连续区间。

## 推理

```powershell
python predict_ab1.py `
  --ab1 data/raw/example.ab1 `
  --model logs_ab1_resnet50_enhanced/best.pth `
  --device cpu
```

输出：

```text
backbone=resnet50
bases=...
start=...
end=...
kept_bases=...
boundary_fallback=False
trimmed_sequence=...
```
