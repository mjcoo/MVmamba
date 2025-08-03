import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_curve, precision_recall_curve
import numpy as np
import os
import math

# 读取 CSV 文件
data = pd.read_csv('./test_result/best_test_predictions.csv')

# 提取真实标签和预测概率
true_labels = data['true_label']
predictions = data['prediction']

# 计算最佳阈值
fpr, tpr, thresholds = roc_curve(true_labels, predictions, pos_label=1)
youden_scores = tpr - fpr
best_threshold_index = np.argmax(youden_scores)
best_threshold = thresholds[best_threshold_index]

# 使用最佳阈值将预测概率转换为二元预测
binary_predictions = np.where(predictions >= best_threshold, 1, 0)

# 计算各项指标
acc = accuracy_score(true_labels, binary_predictions)
auc = roc_auc_score(true_labels, predictions)
aupr = average_precision_score(true_labels, predictions)
f1 = f1_score(true_labels, binary_predictions)
precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
mcc = matthews_corrcoef(true_labels, binary_predictions)

# 计算混淆矩阵元素
tp = np.sum((true_labels == 1) & (binary_predictions == 1))
tn = np.sum((true_labels == 0) & (binary_predictions == 0))
fp = np.sum((true_labels == 0) & (binary_predictions == 1))
fn = np.sum((true_labels == 1) & (binary_predictions == 0))

# 计算其他指标
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
fdr = fp / (fp + tp) if (fp + tp) != 0 else 0
total = len(true_labels)

# 创建一个包含所有指标的 Series
metrics_series = pd.Series({
    'Type': 'GateWaveBimamba',
    'Threshold': best_threshold,  # 添加阈值列
    'AUC': auc,
    'AUPR': aupr,
    'ACC': acc,
    'Recall': recall,
    'Precision': precision,
    'Specificity': specificity,
    'F1': f1,
    'MCC': mcc,
    'FPR': fpr,
    'FDR': fdr,
    'TN': tn,
    'FP': fp,
    'FN': fn,
    'TP': tp,
    'Total': total
})

# 将 Series 转换为 DataFrame 并转置
metrics_df = pd.DataFrame(metrics_series).T

file_path = './test_result/metrics_results.csv'
if not os.path.exists(file_path):
    metrics_df.to_csv(file_path, header=True, index=False)
else:
    metrics_df.to_csv(file_path, header=False, mode='a', index=False)

print("指标计算完成，结果已保存到 metrics_results.csv 文件中。")
print(f"最佳阈值为: {best_threshold:.3f}")