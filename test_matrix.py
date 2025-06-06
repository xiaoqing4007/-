import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
 
classes0 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
classes = ["Praise", "Correction", "Prompts", "Other"]
# 生成数据集的GT标签
# gt_labels = np.zeros(120).reshape(4, -1)
# for i in range(4):
#     gt_labels[i] = i
# print("gt_labels.shape0 : {}".format(gt_labels.shape)) #(10, 100)
# gt_labels = gt_labels.reshape(1, -1).squeeze()
# print("gt_labels.shape : {}".format(gt_labels.shape)) #(1000,)
# print("gt_labels : {}".format(gt_labels[::]))
 
# # 生成数据集的预测标签
# pred_labels = np.zeros(120).reshape(4, -1)
# for i in range(4):
#     # 标签生成规则：对于真值类别编号为i的数据，生成的预测类别编号为[0, i-1]之间的随机值
#     # 这样生成的预测准确率从0到9逐渐递减
#     pred_labels[i] = np.random.randint(0, i + 1, 30)
# pred_labels = pred_labels.reshape(1, -1).squeeze()
# print("pred_labels.shape : {}".format(pred_labels.shape))
# print("pred_labels : {}".format(pred_labels[::]))
 
# 使用sklearn工具中confusion_matrix方法计算混淆矩阵
gt_labels_train = [1,1,1,3,4,3,1,4,2,4,3,1,4,3,4,3,3,1,4,1,
             4,2,4,3,3,3,3,1,4,1,4,3,4,3,3,1,4,3,1,4,
             3,3,3,3,4,3,3,3,1,4,2,4,1,4,3,4,3,3,1,4,
             1,4,3,4,3,3,2,4,3,3,3,3,1,4,1,4,3,3,3,3,
             4,3,3,2,4,3,1,4,3,3,1,4,4,3,4,3,3,2,4,3,
             1,4,1,4,3,4,3,3,3,4,3,1,4,2,4,3,1,4]
pred_labels_train = [1,1,1,3,4,3,1,4,2,4,3,1,4,3,4,3,3,1,4,1,
             4,2,4,3,3,3,3,1,4,1,4,3,4,3,3,1,4,3,1,4,
             3,3,3,3,4,3,3,3,1,4,2,4,1,4,3,4,3,3,1,4,
             1,4,3,4,3,3,2,4,3,3,3,3,1,4,1,4,3,3,3,3,
             4,3,3,2,4,3,1,4,3,3,1,4,4,3,4,3,3,2,4,3,
             1,4,1,4,3,4,3,3,3,4,3,1,4,2,4,3,1,4]
gt_labels =   [1,1,1,3,4,3,1,4,2,4,3,1,4,3,4,3,3,1,4,1,4,2,4,3,3,3,3,1,4,1,4,3]

# ACC=53.1  17/32  H     
                
pred_labels_H = [1,1,1,3,4,4,1,4,4,4,3,3,3,3,4,4,1,4,3,1,3,2,1,3,1,4,4,1,3,1,3,3]
# ACC=56.3  18/32  GAM(LH)+H        

pred_labels_GAM_LH_H = [1,1,1,3,4,3,1,4,2,4,2,1,3,3,4,4,1,4,4,4,3,2,1,3,1,4,4,1,3,1,3,4]
# ACC=53.1  17/32  H+LDA
                    # [1,1,1,3,4,3,1,4,2,4,3,1,4,3,4,3,3,1,4,1,4,2,4,3,3,3,3,1,4,1,4,3]          
pred_labels_H_LDA = [1,4,1,3,4,2,1,4,2,4,3,4,3,3,3,2,3,1,1,1,3,2,3,2,3,3,2,4,3,1,1,4]

# ACC=59.4  19/32  (GAM(LH)+H)+LDA: IN
pred_labels_GAM_LH_H_LDA = [1,1,1,4,4,3,1,4,2,3,4,1,3,3,4,3,4,1,1,4,1,4,2,3,3,3,3,1,1,4,1,3]


gt_labels = np.array(gt_labels)
pred_labels_H = np.array(pred_labels_H)
pred_labels_GAM_LH_H = np.array(pred_labels_GAM_LH_H)
pred_labels_H_LDA = np.array(pred_labels_H_LDA)
pred_labels_GAM_LH_H_LDA = np.array(pred_labels_GAM_LH_H_LDA)
print("gt_labels.shape : {}".format(gt_labels.shape))
print("pred_labels.shape : {}".format(pred_labels_GAM_LH_H_LDA.shape))

confusion_mat1 = confusion_matrix(gt_labels, pred_labels_H)
confusion_mat2 = confusion_matrix(gt_labels, pred_labels_GAM_LH_H)
confusion_mat3 = confusion_matrix(gt_labels, pred_labels_H_LDA)
confusion_mat4 = confusion_matrix(gt_labels, pred_labels_GAM_LH_H_LDA)

print("confusion_mat.shape : {}".format(confusion_mat1.shape))
print("confusion_mat : {}".format(confusion_mat1))
 
# 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat1, display_labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat2, display_labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat3, display_labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat4, display_labels=classes)
fig, ax = plt.subplots(figsize=(8.5, 6))  # Adjust figure size as needed
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)

# Set font size for tick labels
ax.tick_params(axis='both', labelsize=16)

disp.plot(
    include_values=True,            # 混淆矩阵每个单元格上显示具体数值
    cmap="Blues",#"viridis",                 # 
    ax=ax,                        # 同上
    colorbar=True,
    xticks_rotation="horizontal",   # 同上
    values_format="d"               # 显示的数值格式
)
plt.title("H",fontsize=14)
plt.title("GAM(LH)+H",fontsize=14)
plt.title("H+LDA",fontsize=14)
plt.title("(GAM(LH)+H)+LDA",fontsize=14)

plt.show()



# from sklearn.metrics import precision_score

# y_pred = pred_labels_H
# y_pred = pred_labels_GAM_LH_H
# y_pred = pred_labels_H_LDA
# y_pred = pred_labels_GAM_LH_H_LDA
# print(precision_score(gt_labels, y_pred, average='macro'))  # 0.638636363636

# print(precision_score(gt_labels, y_pred, average='micro'))  # 0.53125
# print(precision_score(gt_labels, y_pred, average='weighted'))  # 0.54062 F1 weight
# print(precision_score(gt_labels, y_pred, average=None))  # P:[0.7        1.         0.45454545 0.4       ]
