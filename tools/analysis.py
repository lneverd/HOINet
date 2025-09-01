import re
import matplotlib.pyplot as plt

# 解析日志文件
log_file_path = "../exp/h2o/log2.txt"

# 正则表达式匹配训练和评估日志
train_pattern = re.compile(r"training: epoch: (\d+), loss: ([\d.]+), top1: ([\d.]+)%")
eval_pattern = re.compile(r"evaluating: loss: ([\d.]+), top1: ([\d.]+)%, best_acc: ([\d.]+)%")

epochs = []
train_losses = []
train_top1s = []
eval_epochs = []
eval_losses = []
eval_top1s = []
best_accs = []

with open(log_file_path, "r") as f:
    for line in f:
        train_match = train_pattern.search(line)
        eval_match = eval_pattern.search(line)

        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            top1 = float(train_match.group(3))
            epochs.append(epoch)
            train_losses.append(loss)
            train_top1s.append(top1)

        if eval_match:
            loss = float(eval_match.group(1))
            top1 = float(eval_match.group(2))
            best_acc = float(eval_match.group(3))
            eval_epochs.append(epochs[-1])  # 评估发生在训练后的同一epoch
            eval_losses.append(loss)
            eval_top1s.append(top1)
            best_accs.append(best_acc)

# 绘制训练损失和Top1准确率
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="tab:red")
ax1.plot(epochs, train_losses, "r-", label="Train Loss")
ax1.plot(eval_epochs, eval_losses, "r--", label="Eval Loss")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Top-1 Accuracy (%)", color="tab:blue")
ax2.plot(epochs, train_top1s, "b-", label="Train Top-1")
ax2.plot(eval_epochs, best_accs, "b--", label="Best Eval Top-1")
ax2.tick_params(axis="y", labelcolor="tab:blue")

fig.tight_layout()
plt.title("Training Progress")
plt.legend(loc="lower right")
plt.show()
print('best_acc',max(best_accs))
print('train_top1',max(train_top1s))
print('eval_top1',max(eval_top1s))