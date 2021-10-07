import time, os
from pynvml import *
from subprocess import Popen
import numpy as np
nvmlInit()
import pandas as pd

def run_command(cmd, minmem=2,use_env_variable=True, admissible_gpus=[1],sleep=60):
    sufficient_memory = False
    gpu_idx=0

    while not sufficient_memory:
        time.sleep(sleep)
        # Check free memory
        info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        free_0 = info.free/ 1024 / 1024 / 1024 if 0 in admissible_gpus else 0
        info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(1))
        free_1 = info.free / 1024 / 1024 / 1024 if 1 in admissible_gpus else 0
        if not use_env_variable: #safe mode
            sufficient_memory = np.minimum(free_0,free_1) >=minmem  # 4.5 Gb
        else:
            sufficient_memory = np.maximum(free_0, free_1) >= minmem  # 4.5 Gb
        gpu_idx = np.argmax([free_0,free_1])
        # if not sufficient_memory:
        #     time.sleep(60)
    if use_env_variable:
        # os.system('CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) +cmd)
        proc = Popen(['CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) + cmd.format(0)], shell=True,
                     stdin=None, stdout=None, stderr=None, close_fds=True)
        print('CUDA_VISIBLE_DEVICES="{}" '.format(gpu_idx) + cmd.format(0))
    else:
        os.system(cmd.format(gpu_idx))

import sys
sys.path.append("../")


dataset = 'MIBI2CH'
# dataset = 'MIBI2CH' #100,150,250

dataset = 'Vectra_2CH' #200,300
# scribbles_list = ['150','200']
# scribbles_list = ['150','200']
scribbles_list = ['150']
# scribbles_list = ['200']
# scribbles_list = ['300']
scribbles_list = ['100']



dataset = 'cellpose'
scribbles_list = ['100','200']
scribbles_list = ['200','300']


dataset = 'MIBI1CH'
scribbles_list = ['100', '200']

dataset_list = ['MIBI1CH_Bladder', 'MIBI1CH_Lung']
scribbles_list = ['200','400']


# scribbles_list = ['100']

file_bash_name = dataset+'_bash.sh'
file_bash_name = dataset+'_bash.sh'


model_name_prefix_list=['DS_2tasks_base32depth4relu_adam5e4_mcdrop1e4_nsave5_',
                        'DS_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_']
ubase_list = [32,64]

model_name_prefix_list=['DS_2tasks_base32depth4relu_adam5e4_mcdrop1e4_nsave5_']
ubase_list = [32]

# model_name_prefix = 'DS_2tasks_base64depth4relu_adam5e4_gclip1_' #Todo: ! RE RUN THIS ONE!!
# model_name_prefix = 'DS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_'
# model_name_prefix = 'DS_2tasks_base64depth4relu_adam5e4_nsave5_'
# model_name_prefix = 'DS_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_'
# model_name_prefix = 'DS_2tasks_base64depth4relu_adam5e4_nsave5_'

mcdrop = True
load = False
train = True
evaluation = False
nsave = 5
reset_optim = True

optim = 'adam' #RMSprop
lr=5e-4
optim_regw = 1e-4
# ubase = 64
udepth = 4
activation = 'relu'
batchnorm = False

epochs=400
batch = 64
seed_list=[42,43,44]
seed_list=[42,43,44]
saveout = False
gradclip = 0
gpu = 1


# weights_dic = {'02505':[0.25, 0.25, 0.5],
               # '04501':[0.45, 0.45, 0.1],
               # '00509': [0.05, 0.05, 0.9]} #wfore, wback, wrec

weights_dic = {'04501':[0.45, 0.45, 0.1],
               '02505': [0.25, 0.25, 0.5]} #wfore, wback, wrec

weights_dic = {'04501':[0.45, 0.45, 0.1]} #wfore, wback, wrec
# weights_dic = {'02505':[0.25, 0.25, 0.5],
               # '00509': [0.05, 0.05, 0.9]} #wfore, wback, wrec
# weights_dic = {'00509': [0.05, 0.05, 0.9]} #wfore, wback, wrec
losses_dic = {'segCErecL2':['CE','L2']}

rows_model = []
basedir_root = '/data/natalia/models/'
ix_model = 0

file_bash_name = 'Denoiseg_bash.sh'
with open(file_bash_name, 'w') as f:
    str_root = 'basedir_root="{}"'.format(basedir_root)
    f.write(str_root + '\n\n\n')

    for seed in seed_list:
        if seed == 42:
            saveout = True
        else:
            saveout = False

        for dataset in dataset_list:

            ix_model = 0
            for model_name_prefix in model_name_prefix_list:
                ubase = ubase_list[ix_model]
                ix_model += 1
                if ubase == 128:
                    batch = 32
# with open(file_bash_name,'w') as f:
#
#     str_root = 'basedir_root="{}"'.format(basedir_root)
#     f.write(str_root + '\n\n\n')
#
#     for model_name_prefix in model_name_prefix_list:
#         ubase = ubase_list[ix_model]
#         ix_model += 1
#         if ubase == 128:
#             batch = 32
#
#         for seed in seed_list:
#
#             if seed == 42:
#                 saveout = True
#             else:
#                 False
                for scribbles in scribbles_list:

                    basedir_local = dataset + '/s' + scribbles +'/DenoiSeg/'
                    basedir = basedir_root + basedir_local

                    for loss_key in losses_dic.keys():
                        for weights_key in weights_dic.keys():
                            loss_list = losses_dic[loss_key]
                            weights_list = weights_dic[weights_key]

                            out_file_ext = dataset + '_' + model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed) + '_verbose'
                            model_name = model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed)

                            cmd = 'python main_denoiseg.py --basedir="{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} '.format(basedir,dataset, model_name,saveout,scribbles)
                            # cmd = 'python main_denoiseg.py --basedir={}"{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} --gpu={}'.format('$basedir_root',basedir_local, dataset, model_name,saveout,scribbles,gpu)


                            cmd = cmd + ' --optim_regw={} --optim="{}" --lr={} --gradclip={} --seed={} --train={} --evaluation={}'.format(optim_regw, optim, lr,gradclip,seed,train,evaluation)
                            cmd = cmd + ' --udepth="{}" --ubase="{}" --activation="{}" --batchnorm={}'.format(udepth,ubase,activation,batchnorm)
                            cmd = cmd + ' --seg_loss="{}" --rec_loss="{}" --mcdrop={} --nsaves={} '.format(loss_list[0], loss_list[1],mcdrop, nsave)
                            cmd = cmd + ' --wfore={} --wback={} --wrec={}'.format(weights_list[0], weights_list[1], weights_list[2])
                            cmd = cmd + ' --epochs={} --batch={} --load={} > {}.txt'.format(epochs,batch,load,out_file_ext)

                            rows_model.append(basedir_local + model_name + '/')

                            if ubase == 32:
                                run_command(cmd, minmem=5.5, use_env_variable=True, admissible_gpus=[0,1], sleep=60)
                            else:
                                run_command(cmd, minmem=8, use_env_variable=True, admissible_gpus=[0,1], sleep=60)

                            f.write(cmd + '\n\n\n')
                        f.write('\n\n\n')
                    f.write('\n\n\n')

pd_model = pd.DataFrame(data = rows_model, columns=['model_path'])
pd_model.to_csv('model_path.csv',index = 0)