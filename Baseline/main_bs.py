import sys
import pandas as pd
import numpy as np
import os
import argparse

sys.path.append("../")

from distutils.util import strtobool
from general.utils import mkdir,model_params_load,load_json


cparser = argparse.ArgumentParser()

## General specs
cparser.add_argument('--gpu', action='store', default=0, type=int, help='gpu')
cparser.add_argument('--model_name', action='store', default='Denoiseg', type=str, help='model_name')
cparser.add_argument('--basedir', action='store', default='/data/natalia/models/DenoiSeg/', type=str, help='basedir for internal model save')
cparser.add_argument('--data_dir', action='store', default='/nadeem_lab/Gunjan/data/impartial/', type=str, help='base data dir')
cparser.add_argument('--seed', action='store', default=42, type=int, help='randomizer seed')
cparser.add_argument('--saveout', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: batchnorm')
cparser.add_argument('--load', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: batchnorm')
cparser.add_argument('--train', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: validation stopper')
cparser.add_argument('--multiple_components', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: multiple_components')
cparser.add_argument('--evaluation', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: validation stopper')

## Dataset
cparser.add_argument('--dataset', action='store', default='MIBI2CH', type=str, help='dataset')
cparser.add_argument('--scribbles', action='store', default='150', type=str, help='scribbles tag')

## Network
cparser.add_argument('--activation', action='store', default='relu', type=str, help='activation')
cparser.add_argument('--batchnorm', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: batchnorm')
cparser.add_argument('--udepth', action='store', default=4, type=int, help='unet depth')
cparser.add_argument('--ubase', action='store', default=32, type=int, help='unet fist level number of filters')
cparser.add_argument('--drop_encdec', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: dropout encoder decoder')
cparser.add_argument('--drop_lastconv', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: dropout last conv layer')

##Patch sampling
cparser.add_argument('--patch', action='store', default=128, type=int, help='squre patch size')
cparser.add_argument('--shift_crop', action='store', default=32, type=int, help='shift_crop')
cparser.add_argument('--p_scribble', action='store', default=0.6, type=float, help='probability patch center on scribble ')
cparser.add_argument('--normstd', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: apply standarization to patches')

cparser.add_argument('--min_npatch_image', action='store', default=6, type=int, help='min_npatch_image')
cparser.add_argument('--npatches_epoch', action='store', default=4096, type=int, help='npatches_epoch')
cparser.add_argument('--batch', action='store', default=64, type=int, help='batchsize')

cparser.add_argument('--size_window', action='store', default=10, type=int, help='square blind spot size_window size')
cparser.add_argument('--ratio', action='store', default=0.95, type=float, help='1-ratio proportion of blind spots in patch ')

##Losses
cparser.add_argument('--seg_loss', action='store', default='CE', type=str, help='Segmentation loss')
cparser.add_argument('--wfore', action='store', default=0.25, type=float, help='weight to seg foreground objective')
cparser.add_argument('--wback', action='store', default=0.25, type=float, help='weight to seg background objective')

## Optimizer
cparser.add_argument('--optim_regw', action='store', default=0, type=float, help='regularization weight')
cparser.add_argument('--lr', action='store', default=5e-4, type=float, help='learners learning rate ')
cparser.add_argument('--optim', action='store', default='adam', type=str, help='Learners optimizer')
cparser.add_argument('--valstop', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: validation stopper')
cparser.add_argument('--patience', action='store', default=5, type=int, help='no improvement worst loss patience')
cparser.add_argument('--warmup', action='store', default=50, type=int, help='minimum number of training epochs, no stop is allowed')
cparser.add_argument('--epochs', action='store', default=100, type=int, help='epochs')
cparser.add_argument('--gradclip', action='store', default=0, type=float, help='0 means no clipping')

##Checkpoint ensembles
cparser.add_argument('--nsaves', action='store', default=1, type=int, help='nsaves model') #how many cycles where we reset optimizer and resample training patches
cparser.add_argument('--reset_optim', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: reset optimizer') #reset optimizer between cycles
cparser.add_argument('--nepochs_sample_patches', action='store', default=10, type=int, help='minimum number of epochs before resampling image patches')

##MCdropout
cparser.add_argument('--mcdrop', action='store', default=False, type=lambda x: bool(strtobool(x)), help='boolean: mc_dropout?')
cparser.add_argument('--mcdrop_iter', action='store', default=10, type=int, help='mcdropout iterations during inference')


cparser = cparser.parse_args()

if __name__== '__main__':

    data_dir = os.path.join(cparser.data_dir, cparser.dataset) #moved outside of 'if'
    pd_files = pd.read_csv(os.path.join(data_dir, 'files.csv'), index_col=0)

    if cparser.nepochs_sample_patches == 0:
        cparser.nepochs_sample_patches = 10


    if cparser.dataset == 'Vectra_2CH':
        
        scribble_fname = 'files_2tasks1x2classes_3images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        n_channels = 2
        classification_tasks = {'0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]},
                                '1': {'classes': 2, 'rec_channels': [1], 'ncomponents': [1, 1, 2]}}
        

    if cparser.dataset == 'MIBI2CH':

        scribble_fname = 'files_2tasks1x2classes_3images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        n_channels = 2

        if cparser.multiple_components:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2], 'rec_channels': [0]},
                                    '1': {'classes': 2, 'ncomponents': [1, 1, 2], 'rec_channels': [1]}}
        else:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [1, 1], 'rec_channels': [0]},
                                    '1': {'classes': 2, 'ncomponents': [1, 1, 1], 'rec_channels': [1]}}

    if cparser.dataset == 'MIBI1CH':
        scribble_fname = 'files_1task1class_4images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        n_channels = 1

        if cparser.multiple_components:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2], 'rec_channels': [0]}}
        else:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [1, 1], 'rec_channels': [0]}}


    if cparser.dataset == 'MIBI1CH_Bladder':
        scribble_fname = 'files_1task1class_10images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        n_channels = 1

        if cparser.multiple_components:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2], 'rec_channels': [0]}}
        else:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [1, 1], 'rec_channels': [0]}}


    if cparser.dataset == 'MIBI1CH_Lung':
        scribble_fname = 'files_1task1class_10images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        n_channels = 1

        classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2], 'rec_channels': [0]}}


    if cparser.dataset == 'cellpose':
        scribble_fname = 'files_1task1class_10images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles)

        pd_files = pd.read_csv(os.path.join(data_dir, 'files.csv'))

        n_channels = 2

        if cparser.multiple_components:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [2, 2], 'rec_channels': [0,1]}}
        else:
            classification_tasks = {'0': {'classes': 1, 'ncomponents': [1, 1], 'rec_channels': [0,1]}}


    print('loaded :', files_scribbles)
    print('Total images  train: ', len(pd_files_scribbles),'; test: ', len(pd_files)-len(pd_files_scribbles))

    for task in classification_tasks.keys():
        # get list of corresponding gt indexes
        ix_labels_list = pd_files_scribbles['gt_index_task' + task].values[0]
        if ',' in ix_labels_list[1:-1]:
            ix_labels_list = ix_labels_list[1:-1].split(',')
        else:
            ix_labels_list = [ix_labels_list[1:-1]]
        classification_tasks[task]['ix_gt_labels'] = ix_labels_list

    #------------------------- Config file --------------------------------#

    from Baseline.bs_classes import Config
    patch_size = (cparser.patch, cparser.patch)
    size_window = (cparser.size_window, cparser.size_window)

    weight_objectives = {'seg_fore':cparser.wfore,'seg_back':cparser.wback}

    npatch_image_sampler = np.maximum(int(cparser.npatches_epoch/len(pd_files_scribbles)), cparser.min_npatch_image)
    nepochs_sample_patches = np.maximum(int(cparser.warmup/cparser.nsaves), cparser.nepochs_sample_patches)


    config = Config(basedir=cparser.basedir,
                    model_name=cparser.model_name,
                    n_channels=n_channels, #network params
                    activation = cparser.activation,
                    batchnorm = cparser.batchnorm,
                    unet_depth = cparser.udepth,
                    unet_base = cparser.ubase,
                    normstd = cparser.normstd,

                    drop_last_conv = cparser.drop_lastconv,
                    drop_encoder_decoder=cparser.drop_encdec,

                    patch_size=patch_size, #patch sampling
                    shift_crop=cparser.shift_crop,
                    p_scribble_crop=cparser.p_scribble,
                    npatches_epoch=cparser.npatches_epoch,
                    npatch_image_sampler = npatch_image_sampler,
                    nepochs_sample_patches = nepochs_sample_patches,
                    BATCH_SIZE = cparser.batch,
                    size_window = size_window, #blind spot sampler
                    ratio = cparser.ratio,
                    seg_loss = cparser.seg_loss, #loss
                    classification_tasks = classification_tasks,
                    weight_objectives = weight_objectives,
                    optimizer=cparser.optim, #optimizer
                    optim_weight_decay = cparser.optim_regw,
                    LEARNING_RATE= cparser.lr,
                    max_grad_clip=cparser.gradclip,
                    val_stopper = cparser.valstop, #Training
                    patience = cparser.patience,
                    warmup_epochs = cparser.warmup,
                    EPOCHS=cparser.epochs,
                    seed=cparser.seed,
                    GPU_ID=cparser.gpu,
                    nsaves = cparser.nsaves,
                    reset_optim=cparser.reset_optim,

                    MCdrop = cparser.mcdrop,
                    MCdrop_it=cparser.mcdrop_iter)

    mkdir(config.basedir)
    mkdir(os.path.join(config.basedir, config.model_name))

    # ------------------------- Model Setup --------------------------------#
    from Baseline.bs_classes import BaselineModel
    im_model = BaselineModel(config)

    model_output_dir = os.path.join(im_model.config.basedir, im_model.config.model_name)

    if cparser.load:
        _path = os.path.join(model_output_dir, im_model.config.best_model)
        if os.path.exists(_path):
            print(' Loading : ', _path)
            model_params_load(_path, im_model.model, im_model.optimizer, im_model.config.DEVICE)

    # ------------------------- Training --------------------------------#
    
    model_output_hist_path = os.path.join(model_output_dir, 'history.json')
    model_output_pd_summary_path = os.path.join(model_output_dir, 'pd_summary_results.csv')
    
    if cparser.train:

        ## load dataloader
        im_model.load_dataloaders(data_dir, pd_files_scribbles)

        history = im_model.train()

        print(' Saving .... ')
        im_model.config.save_json()
        for key in history.keys():
            history[key] = np.array(history[key]).tolist()

        from general.utils import save_json

        im_model.config.save_json()
        save_json(history, model_output_hist_path)
        print('history file saved on : ', model_output_hist_path)

    else:
        history = load_json(model_output_hist_path)


    # ------------------------- Evaluation --------------------------------#
    if cparser.evaluation:
        pd_summary = im_model.data_performance_evaluation(pd_files, data_dir, saveout=cparser.saveout, plot=False, default_ensembles=True)

        pd_summary.to_csv(model_output_pd_summary_path, index=0)
        print('Evaluation csv saved on : ', model_output_pd_summary_path)
