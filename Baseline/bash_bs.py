import time, os
from pynvml import *
from subprocess import Popen
import numpy as np
nvmlInit()

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
# dataset = 'Vectra_2CH'

# scribbles_list = ['150','200']
scribbles_list = ['150','200']

saveout = True

file_bash_name = dataset+'_bash.sh'
model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_'
# model_name_prefix = 'BS_2tasks_base64depth4relu_adam5e4_gclip10_nsave5_ratio1_'

load = False
train = True
nsave = 6
ratio = 0.95

optim = 'adam' #RMSprop
lr=5e-4
regweight = 0
ubase = 64
udepth = 4
activation = 'relu'
batchnorm = False

epochs=400
batch = 64
if ubase == 128:
    batch = 32
seed_list=[42,43,44]
gradclip = 10

gpu = 1
weights_dic = {'0500': [0.5, 0.5]} #wfore, wback, wrec
losses_dic = {'segCE':['CE']}

with open(file_bash_name,'w') as f:
    for seed in seed_list:
        for scribbles in scribbles_list:
            basedir = '/data/natalia/models/' + dataset + '/s'+scribbles + '/Baseline/'

            for loss_key in losses_dic.keys():
                for weights_key in weights_dic.keys():
                    loss_list = losses_dic[loss_key]
                    weights_list = weights_dic[weights_key]

                    out_file_ext = dataset + '_' + model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed) + '_verbose'
                    model_name = model_name_prefix + loss_key + '_w'+ weights_key +'_seed' + str(seed)

                    # cmd = 'python main_bs.py --basedir="{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} '.format(basedir,dataset, model_name,saveout,scribbles)
                    cmd = 'python main_bs.py --basedir="{}" --dataset="{}" --model_name="{}" --saveout={} --scribbles={} --gpu={}'.format(basedir, dataset, model_name,saveout,scribbles,gpu)


                    cmd = cmd + ' --optim_regw={} --optim="{}" --lr={} --gradclip={} --seed={} --train={} '.format(regweight, optim, lr,gradclip,seed,train)
                    cmd = cmd + ' --udepth="{}" --ubase="{}" --activation="{}" --batchnorm={}'.format(udepth,ubase,activation,batchnorm)
                    cmd = cmd + ' --seg_loss="{}" --nsaves={} --ratio={} '.format(loss_list[0],nsave,ratio)
                    cmd = cmd + ' --wfore={} --wback={}'.format(weights_list[0], weights_list[1])
                    cmd = cmd + ' --epochs={} --batch={} --load={} > {}.txt'.format(epochs,batch,load,out_file_ext)

                    # run_command(cmd, minmem=7, use_env_variable=True, admissible_gpus=[1], sleep=120)
                    f.write(cmd + '\n\n\n')
                f.write('\n\n\n')
            f.write('\n\n\n')