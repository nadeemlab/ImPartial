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


# dataset = 'Vectra_2CH'
# scribbles_list = ['200','300']
#
# dataset = 'MIBI2CH'
# scribbles_list = ['150','250','100']

dataset = 'cellpose'
scribbles_list = ['200']


model_name_prefix_list=['BS_2tasks_base32depth4relu_adam5e4_mcdrop1e4_nsave5_',
                        'BS_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_']
ubase_list = [32,64]
udepth_list = [4,4]

model_name_prefix_list=['BS_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_',
                        'BS_2tasks_base64depth3relu_adam5e4_mcdrop1e4_nsave5_']
ubase_list = [64,64]
udepth_list = [4,3]


model_name_prefix_list=['BS_2tasks_base64depth3relu_adam5e4_mcdrop1e4_nsave5_']
ubase_list = [64]
udepth_list = [3]

saveout = True

file_bash_name = dataset+'_bash.sh'
# model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_nsave6_'
# model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_'
# model_name_prefix = 'BS_non2v_2tasks_base64depth4relu_adam5e4_nsave5_'
# model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_mcdrop_nsave5_'
# model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_nsave5_'
# ubase = 64


mcdrop = True
load = False
train = True
nsave = 5
ratio = 0.95
multiple_components = False

optim = 'adam' #RMSprop
lr=5e-4
optim_regw = 1e-4


activation = 'relu'
batchnorm = False


epochs=4


seed_list=[42]
gradclip = 0

gpu = 1
weights_dic = {'0500': [0.5, 0.5]} #wfore, wback, wrec
losses_dic = {'segCE':['CE']}




rows_model = []
basedir_root = '/data/natalia/models/'
ix_model = 0

with open(file_bash_name,'w') as f:

    str_root = 'basedir_root="{}"'.format(basedir_root)
    f.write(str_root + '\n\n\n')

    for model_name_prefix in model_name_prefix_list:
        ubase = ubase_list[ix_model]
        udepth = udepth_list[ix_model]
        ix_model += 1
        batch = 64
        if ubase == 128:
            batch = 32


        for seed in seed_list:
            for scribbles in scribbles_list:
                basedir_local = dataset + '/s' + scribbles + '/Baseline/'
                basedir = basedir_root + basedir_local

                for loss_key in losses_dic.keys():
                    for weights_key in weights_dic.keys():
                        loss_list = losses_dic[loss_key]
                        weights_list = weights_dic[weights_key]

                        out_file_ext = dataset + '_' + model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed) + '_verbose'
                        model_name = model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed)

                        cmd = 'python main_bs.py --basedir="{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} '.format(basedir,dataset, model_name,saveout,scribbles)
                        # cmd = 'python main_bs.py --basedir="{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} --gpu={}'.format(basedir, dataset, model_name,saveout,scribbles,gpu)
                        cmd = 'python main_ms.py --basedir={}"{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} --gpu={}'.format('$basedir_root',basedir_local, dataset, model_name,saveout,scribbles,gpu)

                        cmd = cmd + ' --optim_regw={} --optim="{}" --lr={} --gradclip={} --seed={} --train={}'.format(optim_regw, optim, lr,gradclip,seed,train)
                        cmd = cmd + ' --udepth="{}" --ubase="{}" --activation="{}" --batchnorm={}'.format(udepth,ubase,activation,batchnorm)
                        cmd = cmd + ' --seg_loss="{}" --nsaves={} --mcdrop={} --ratio={}'.format(loss_list[0],nsave,mcdrop,ratio)
                        cmd = cmd + ' --wfore={} --wback={}'.format(weights_list[0], weights_list[1])
                        cmd = cmd + ' --epochs={} --batch={} --load={} --multiple_components={} > {}.txt'.format(epochs, batch, load,multiple_components, out_file_ext)

                        rows_model.append(basedir_local + model_name + '/')
                        if ubase == 32:
                            run_command(cmd, minmem=5.5, use_env_variable=True, admissible_gpus=[1], sleep=60)
                        else:
                            run_command(cmd, minmem=8, use_env_variable=True, admissible_gpus=[1], sleep=60)
                        f.write(cmd + '\n\n\n')
                    f.write('\n\n\n')
                f.write('\n\n\n')

pd_model = pd.DataFrame(data = rows_model, columns=['model_path'])
pd_model.to_csv('model_path.csv',index = 0)