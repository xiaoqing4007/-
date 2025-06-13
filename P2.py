
# ***************np.prod()****************

import numpy as np

arr = np.array([1, 2, 3, 4])
print(np.prod(arr))  # 输出: 24

arr2d = np.array([[1, 2], [3, 4]])

print(np.prod(arr2d))           # 所有元素相乘: 24
print(np.prod(arr2d, axis=0))   # 每列相乘: [3 8]
print(np.prod(arr2d, axis=1))   # 每行相乘: [2 12]

############ dot multiply, get Dim
obs_dim = int(np.prod(env.observation_space.shape))


##############Norm#############
self._obs_mean = np.mean(obs_batch, axis=0)
self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

########## layer Normlize##########
def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output

########---------tf2rl_git/tools/ ----------########
########################################################
#########  1. check_pkl.py
#show pkl files content

import joblib
f = "pkl path"
data = joblib.load(open('pkl path', 'rb'))
f.close()
data.keys()
data.get('obs').shape
data.get('obs')[:4]
########################################################
#########  2. data_process_test_trans_feature.py##########
import pandas as pd
data_input = pd.read_csv(data_PATH)
        # index = data_input['index'][:].to_list()
        # print(index[-4:])
        action = data_input['action sentence_raw']
######Add path to current path
sys.path.append(r'/home/xiaoqing007/project/tf2rl/tf2rl_git/data/')
import config1   #config1.py
TASK = config1.TASK

# write python dict to a file
import pickle
            output = open(data_pkl_PATH, 'wb')
            pickle.dump(data, output)
            output.close()

            # read python dict back from the file
            pkl_file = open(data_pkl_PATH, 'rb')
            mydict2 = pickle.load(pkl_file)
            pkl_file.close()
########save new col to csv file######
def save_to_train_test_result_csv(path, v1, v2, v3 ):
    name_r=['task_test','task_choose','result']
    data_r_show = list(zip(v1,v2, v3))
    test_r=pd.DataFrame(columns=name_r,data=data_r_show)
    test_r.to_csv(path,encoding='gbk')

import csv
    with open(new_file_path3,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["train_reward","train_td_error", "test_reward"]
        csv_write.writerow(csv_head)

########## COS, KL, DTW 
        if type=='cos':
            sim = cos_sim_value(vector_a, vector_b)
        elif type=='KL':
            sim = KL_sim_value(vector_a, vector_b)
        elif type=='DTW':  
            sim = DTW_sim_value(vector_a, vector_b)

# 标准化
def Standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    result = (data - mu) / sigma
    print('-----', result)
    return result
def Normalization_np(data):
    temp = np.linalg.norm(np.array(data), keepdims=True)
    # print('=========temp:',temp)
    return np.array(data)/ temp

########################################################
######### 3. folder_build.py#########

#########Add folder
import os
def Add_folder(path, task, new_folder_name):
    new_folder_path = path + str(task) + '/' + In_Feature + '/' + str(0) + '/' + str(Head) +'H-' + str(Layer) + 'L/' + new_folder_name
    # os.mkdir(new_folder_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
import shutil
def remove_folder(path):
    # os.remove(path)  #rm file
    # os.removedirs(path)   #删除空文件夹
    # os.rmdir(path)    #删除空文件夹
    shutil.rmtree(path)    #递归删除文件夹，即：删除非空文件夹
    # remove_file(Current_path,16, 'ex', Head,Layer)
os.rename(old_folder_path,new_folder_path)

#get aqure sum root        , Fan shu
def Norm_vector(vector_a):   
    vector_a_norm = np.linalg.norm(vector_a)
    print('-----', vector_a_norm)
    return vector_a_norm
########################################################
######### 4. pytorch_test_gpu.py##########
import torch
# env 测试     tf20成功  tf2rl tf2rl2
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)

print(torch.cuda.is_available())    #env：tf2rl ubuntu, tf2rl tf2rl2

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.current_device())
    print('%d GPU is available.'%torch.cuda.device_count())
    print('We will use the GPU: ',torch.cuda.get_device_name(0))
else:
    print('No GPU available,use CPU instead.')
    device = torch.device('cpu')



########################################################
######### 5. result_xlsx_build_pc.py###########
    def Init_xlsx_build(path, sheet_name, value):
    try:
        with open(path,mode='r',encoding='utf-8') as ff:
            print("Open xlsx file.")          
    except FileNotFoundError: 
         ....

########################################################
#########    6. tf_test_gpu.py                ##########
# tensorflow GPU test
# 
# 


##############/example/##########################################
#########    7. run_gail_ddpg_TransFeature_pc.py
Inner Loop  DDPG+GAIL
Outer Loop  Reptile

##########################################################
a = [[1,2,5,6,7],[2,2,3,4,5]]
b = [[2,2,3,4,5]]
import scipy.stats
c = scipy.stats.entropy(a,b)
# data = {'obs':[]}
# data['obs'].append(a)
# data['obs'].append(b)
# data['obs'].append(a)
# c = np.concatenate((a, b), axis=0).reshape(2,-1)
print(c)


##########实现 __call__ 方法的类，其对象可以像函数一样使用。
class Greeter:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        print(f"Hello, {self.name}!")

g = Greeter("Alice")
g()  # 等同于 g.__call__()，输出：Hello, Alice!