###########import 用法
from 文件夹名 import 文件夹名
from 文件夹名.python 文件名 import 函数名1，函数名2


############param 用法 in arguments.py
import argparse
def get_args()
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    args = parser.parse_args()
    #####用于判断是否启用 CUDA（GPU 加速） 
    # 如果你没有显式禁用 CUDA（args.no_cuda 为 False）且 CUDA 可用，那就返回 True，表示可以用 GPU
    #not args.no_cuda 表示命令行参数中是否禁用 CUDA 
    #torch.cuda.is_available()
    # torch.cuda.is_available()检查你的计算机是否有可用的 CUDA GPU（比如是否装了显卡驱动、PyTorch CUDA 版本等）。
    args.cuda = not args.no_cuda and torch.cuda.is_available() 
      
    assert args.algo in ['a2c', 'ppo', 'acktr']    
    return args
#断言语句，用来验证程序运行时是否满足某个条件。如果条件不满足，就会抛出 AssertionError 并中断程序

###############使用参数
####读取参数
args = get_args()
params = {}
params['num_mini_batch'] = args.num_mini_batch
###########程序运行
python main.py --env-name "lmnoprefix"

##############读取CVS文件，用pandas, 
######clinic_gen.py
# ctrl+Q toggle 注释

import pandas as pd

df = pd.DataFrame(pd.read_csv(clinic_path, index_col=0))
data_col = df.iloc[:, 5:12]  # 取第2-7列
# row = df.loc[1, :]   # 按数字索引取第2行

clinic["startphrase"] = data_col[:]['action sentence_raw'].values
clinic["pool"] = np.append(data_col[:]['action sentence_raw'].values, [[''],['']]) #将元素添加到数组末尾的函数

#########随机数生成
from random import choice
label = choice(np.random.randint(4,size=4)) #【0 1 2 3】随机4个数 SIZE=4
#########随机种子 种子保证每次运行输出：例如 [2 2 6 1 3]（始终相同）
import numpy as np
params = {'seed': 123}
np.random.seed(params['seed'])
print(np.random.randint(0, 10, size=5))
# 每次运行输出：例如 [2 2 6 1 3]（始终相同）




##########去除重复元素
in_data = [1, 2, 2, 3, 4, 4, 4, 5]
in_data_ = list(set(in_data))
print(in_data_)  # 输出可能是 [1, 2, 3, 4, 5]，但顺序不保证


########发现NaN,for的三种写法
for index, option in enumerate(in_data):
    if option is np.nan:
        print('Find NaN....')
        break
        
for content in in_data:    
for i in range(len(in_data['Options'])):


###########pkl文件 
####读
root_path = os.path.join(self.dataset_dir, 'clinic_train.pkl')
f1_save = open(root_path, 'rb')
clinic_train = pickle.load(f1_save)
clinic_train_raw = clinic_train["startphrase"]

######写！打开读取csv后，写入保存 为pkl文件
import pickle
clinic_data1 = read_clinic_csv(clinic_path1)
clinic_data2 = read_clinic_csv(clinic_path2)
f1_save = open(clinic_pkl_train, 'wb')
f2_save = open(clinic_pkl_test, 'wb')
pickle.dump(clinic_data1, f1_save) #clinic_data1 has the data as dict
pickle.dump(clinic_data2, f2_save)
f1_save.close()
f2_save.close()

########读JSON 模型参数文件
# 把原始bert中的配置参数也导入进来
self.dataset_dir = 'D:\\PHD\\Project3\\tempera\\clinic_bert\\'
self.pretrained_model_dir = os.path.join(self.dataset_dir, "bert_base_uncased_english")
bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
bert_config = BertConfig.from_json_file(bert_config_path)
for key, value in bert_config.__dict__.items():
    self.__dict__[key] = value

##########打开 txt， 没有则创建
filePath = './log/'+ str(args.num_env_steps) +'_'+ params['layer_lmh'] + '_'+ params['fusion_score'] + '.txt'
istxtExist = os.path.exists(filePath)
if not istxtExist:
    # os.makedirs(filePath)
    print_result = open(filePath,'a')#如果有这个文件就打开，如果没有这个文件就创建一个txt文件
else:
    print_result = open(filePath,'a')
    
############查看路径存在否，没有则创建
 file_path = './checkpoints/'+str(args.models)+'_'+str(args.datasets)+'_'+str(args.seed)+'/'
                isExist = os.path.exists(file_path)
                if not isExist:
                    os.makedirs(file_path)   
##########保存 pth 数据
torch.save(current_prompt_embedding_pool, file_path+'current_prompt_embedding_pool.pth')

##############建立多个进程
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!  #建立多个进程，每个进程都会执行main_worker函数
    main()


###########字符型 先转浮点数，再转整数，只取整数部分，不是四舍五入
clinic_train_split = "3.7"
result = int(float(clinic_train_split))
print(result)  # 输出：3

##########四舍五入的用法  .5 的情况会向最接近的偶数靠拢
round(float(2.5))  # 四舍五入 2
math.floor(float(3.7))  # 向下取整 3
math.ceil(float(2.1))   # 向上取整 3


########四舍五入的用法  .5 的情况会向上进位 用decimal
from decimal import Decimal, ROUND_HALF_UP
n = Decimal('2.5')  # 将字符串 '2.5' 转换为高精度 Decimal 对象
rounded = n.quantize(Decimal('1'), rounding=ROUND_HALF_UP)  #表示要保留到“个位数”（即没有小数）。
#decimal 模块提供的一个数据类型，用于进行高精度、小数精确控制的数学计算
print(int(rounded))  # 输出：3



##########不等长 补零 拼接
import torch
from torch.nn.utils.rnn import pad_sequence
# 三个变长序列
seq1 = torch.tensor([1, 2, 3])
seq2 = torch.tensor([4, 5])
seq3 = torch.tensor([6])
# 放入列表
sequences = [seq1, seq2, seq3]
# 填充并设置 batch_first=True
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print(padded)
# 输出     tensor([[1, 2, 3],
        #           [4, 5, 0],
        #           [6, 0, 0]])
        
      
#############Torch ‘cat' 的拼接，所有张量的形状在除拼接维以外的维度上必须一致！
########新行 拼接
import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6]])
result = torch.cat([a, b], dim=0)
print(result)
# tensor([[1, 2],
        # [3, 4],
        # [5, 6]])
        
########新列 拼接
result = torch.cat([a, b], dim=1)
print(result)
# tensor([[1, 2, 5],
        # [3, 4, 6]])

#########torch.stack() 的区别， 新增维度的拼接
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
torch.cat([a, b])      # 1*4→ tensor([1, 2, 3, 4])
torch.stack([a, b])    # 2*2 → tensor([[1, 2],
                       #            [3, 4]])



#########构造一个新的字典 number_dict，
#########其作用是：以 params['label_dict'] 中的每个键为 key，值全部初始化为 0
number_dict = {x:0 for x in params['label_dict'].keys()}  #{0:0, 1:0}


########列表推导式（list comprehension），用于根据索引列表 idxs 从原始句子列表 sentences 中提取特定句子
sentences = ["I love cats.", "Dogs are great.", "Rabbits are cute."]
idxs = [0, 2]
selected_sentences = [sentences[i] for i in idxs]
# ["I love cats.", "Rabbits are cute."]


###########for 循环搭配 zip() 的结构,常用于对齐数据和标签进行迭代训练或处理
for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
    # 在这里执行你想对每对句子和标签做的操作
    print(train_sentence, train_label)
    
for i, (train_sentence, train_label) in enumerate(zip(hundred_train_sentences, hundred_train_labels)):
    print(f"[{i}] {train_sentence} → {train_label}")
    
    
    
##############画散点图
import matplotlib.pyplot as plt
import numpy as np

# x = np.array(['L','M','H','GAM(L)','GAM(M)','GAM(H)','LH','GAM(LH)','GAM(LH)+H','H+LDA','(GAM(LH)+H)+LDA :LN','(GAM(LH)+H)+LDA: IN'])
# y = np.array([21.9,31.3,53.1,28.1,40.6,46.9,37.5,50.0,56.3,53.1,56.3,59.4])
# plt.scatter(x, y, color = 'hotpink')

# x = np.array([1,2,3,4,5,6,7])
# y = np.array([43.8,40.6,53.1,40.6,46.9,46.9,59.4])
# plt.scatter(x, y, color = '#88c999')

plt.xlabel("x - label")
plt.ylabel("F1-weight")
x = np.array(['H','GAM(LH)+H','H+LDA','(GAM(LH)+H)+LDA'])
y = np.array([0.541,0.552,0.553,0.589])
# colors = np.array(["red","green","black","orange","purple","beige","cyan","magenta"])
colors = np.array(["red","green","orange","purple"])
plt.scatter(x, y, c=colors)
# plt.scatter(x, y, color = 'orange')

plt.show()


################画混淆矩阵 confusion matrix 图
test_matrix.py
###########完全二叉树
test_tree.py