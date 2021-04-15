import sys
sys.path.append("../")
import pandas as pd
from general.utils import mkdir,model_params_load
import numpy as np
from dataprocessing.dataloaders import ImageBlindSpotDataset
import os
import argparse
from distutils.util import strtobool

cparser = argparse.ArgumentParser()

## General specs
cparser.add_argument('--gpu', action='store', default=0, type=int,help='gpu')
cparser.add_argument('--model_name', action='store', default='Denoiseg', type=str, help='model_name')
cparser.add_argument('--basedir', action='store', default='/data/natalia/models/DenoiSeg/', type=str,help='basedir for internal model save')
cparser.add_argument('--seed', action='store', default=42, type=int, help='randomizer seed')
cparser.add_argument('--saveout', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: batchnorm')
cparser.add_argument('--load', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: batchnorm')

## Dataset
cparser.add_argument('--dataset', action='store', default='MIBI2CH', type=str,help='dataset')
cparser.add_argument('--scribbles', action='store', default='150', type=str,help='scribbles tag')

## Network
cparser.add_argument('--activation', action='store', default='relu', type=str,help='activation')
cparser.add_argument('--batchnorm', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: batchnorm')
cparser.add_argument('--udepth', action='store', default=4, type=int, help='unet depth')
cparser.add_argument('--ubase', action='store', default=32, type=int, help='unet fist level number of filters')

##Patch sampling
cparser.add_argument('--patch', action='store', default=128, type=int, help='squre patch size')
cparser.add_argument('--shift_crop', action='store', default=32, type=int, help='shift_crop')
cparser.add_argument('--p_scribble', action='store', default=0.6, type=float, help='probability patch center on scribble ')
cparser.add_argument('--normstd', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: apply standarization to patches')

cparser.add_argument('--min_npatch_image', action='store', default=6, type=int, help='min_npatch_image')
cparser.add_argument('--npatches_epoch', action='store', default=1024, type=int, help='npatches_epoch')
cparser.add_argument('--nepochs_sample_patches', action='store', default=0, type=int, help='nepoch_sample_patches')
cparser.add_argument('--batch', action='store', default=64, type=int, help='batchsize')

cparser.add_argument('--size_window', action='store', default=10, type=int, help='square blind spot size_window size')
cparser.add_argument('--ratio', action='store', default=0.95, type=float, help='1-ratio proportion of blind spots in patch ')

##Losses
cparser.add_argument('--seg_loss', action='store', default='CE', type=str,help='Segmentation loss')
cparser.add_argument('--rec_loss', action='store', default='gaussian', type=str,help='Reconstruction loss')
cparser.add_argument('--nfactors', action='store', default=1, type=int, help='nfactors ')
cparser.add_argument('--mean', action='store', default=True,type=lambda x: bool(strtobool(x)),help='fit mean each component')
cparser.add_argument('--std', action='store', default=False,type=lambda x: bool(strtobool(x)),help='fit std each component')

cparser.add_argument('--wfore', action='store', default=0.25, type=float, help='weight to seg foreground objective')
cparser.add_argument('--wback', action='store', default=0.25, type=float, help='weight to seg background objective')
cparser.add_argument('--wrec', action='store', default=0.5, type=float, help='weight to reconstruction objective')

## Optimizer
cparser.add_argument('--optim_regw', action='store', default=0, type=float, help='regularization weight')
cparser.add_argument('--lr', action='store', default=5e-5, type=float, help='learners learning rate ')
cparser.add_argument('--optim', action='store', default='adam', type=str,help='Learners optimizer')
cparser.add_argument('--valstop', action='store', default=True,type=lambda x: bool(strtobool(x)),help='boolean: validation stopper')
cparser.add_argument('--patience', action='store', default=20, type=int, help='no improvement worst loss patience')
cparser.add_argument('--warmup', action='store', default=100, type=int, help='warmup epochs, no stop is allowed')
cparser.add_argument('--epochs', action='store', default=80, type=int, help='epochs')
cparser.add_argument('--gradclip', action='store', default=0, type=float, help='0 means no clipping')
cparser = cparser.parse_args()

if __name__== '__main__':

    if cparser.dataset == 'MIBI2CH_3tasks':

        data_dir = '/data/natalia/intern20/PaperData/MIBI_2channel/'
        # pd_files_scribbles = pd.read_csv(data_dir + 'files_scribbles_train_'+cparser.scribbles+'.csv')
        files_scribbles = data_dir + 'files_2tasks3classes_scribble_train_' + cparser.scribbles + '.csv'
        pd_files_scribbles = pd.read_csv(files_scribbles)

        pd_files = pd.read_csv(data_dir + 'files.csv', index_col=0)
        n_channels = 2
        # classification_tasks = {'0': {'classes': 1, 'rec_channels': [0, 1]},
                                # '1': {'classes': 2, 'rec_channels': [1]}}
        nclasses = 3
        if cparser.nepochs_sample_patches == 0:
            cparser.nepochs_sample_patches = 25

    print('loaded :', files_scribbles)
    print('Total images  train: ', len(pd_files_scribbles),'; test: ', len(pd_files)-len(pd_files_scribbles))

    #------------------------- Config file --------------------------------#
    from ImpartialFactors.imf_classes import ImPartialFactorsConfig
    patch_size = (cparser.patch,cparser.patch)
    size_window = (cparser.size_window,cparser.size_window)
    weight_objectives = {'seg_fore':cparser.wfore,'seg_back':cparser.wback,'rec':cparser.wrec}
    npatch_image_sampler = np.maximum(int(cparser.npatches_epoch/len(pd_files_scribbles)),cparser.min_npatch_image)

    config = ImPartialFactorsConfig(basedir=cparser.basedir,
                            model_name=cparser.model_name,
                            n_channels=n_channels, #network params
                            activation = cparser.activation,
                            batchnorm = cparser.batchnorm,
                            unet_depth = cparser.udepth,
                            unet_base = cparser.ubase,
                            normstd = cparser.normstd,
                            patch_size=patch_size, #patch sampling
                            shift_crop=cparser.shift_crop,
                            p_scribble_crop=cparser.p_scribble,
                            npatches_epoch=cparser.npatches_epoch,
                            npatch_image_sampler = npatch_image_sampler,
                            nepochs_sample_patches = cparser.nepochs_sample_patches,
                            BATCH_SIZE = cparser.batch,
                            size_window = size_window, #blind spot sampler
                            ratio = cparser.ratio,
                            seg_loss = cparser.seg_loss, #loss
                            rec_loss = cparser.rec_loss,
                                    nclasses=nclasses,
                            nfactors = cparser.nfactors,
                            mean=cparser.mean,
                            std=cparser.std,
                            weight_objectives = weight_objectives,
                            optimizer=cparser.optim, #optimizer
                            optim_weight_decay = cparser.optim_regw,
                            LEARNING_RATE= cparser.lr,
                            max_grad_clip = cparser.gradclip,
                            val_stopper = cparser.valstop, #Training
                            patience = cparser.patience,
                            warmup_epochs = cparser.warmup,
                            EPOCHS=cparser.epochs,
                            seed=cparser.seed,
                            GPU_ID=cparser.gpu)

    mkdir(config.basedir)
    mkdir(config.basedir+config.model_name+'/')

    print('---------------- Impartial model config created ----------------------------')
    print('Model directory:', config.basedir + config.model_name + '/')
    print('Config file :')
    print(config)
    print('')


    #------------------------- Dataloaders --------------------------------#
    from dataprocessing.dataloaders import Normalize,RandomFlip,ToTensor
    from torchvision import transforms
    import torch

    ### Dataloaders for training
    transforms_list = []
    if config.normstd:
        transforms_list.append(Normalize(mean=0.5, std=0.5))
    if config.augmentations:
        transforms_list.append(RandomFlip())
    transforms_list.append(ToTensor(dim_data = 3))

    transform_train = transforms.Compose(transforms_list)

    #dataaset train
    dataset_train = ImageBlindSpotDataset(pd_files_scribbles,transform=transform_train,validation = False,
                                             ratio=config.ratio, size_window=config.size_window,
                                             p_scribble_crop = config.p_scribble_crop, shift_crop = config.shift_crop,
                                             patch_size=config.patch_size, npatch_image = config.npatch_image_sampler)
    dataset_train.sample_patches_data(npatches_total = config.npatches_epoch) #sample first epoch patches
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.BATCH_SIZE, shuffle=True,
                                                   num_workers=config.n_workers)



    ### Dataloader for evaluation
    from dataprocessing.dataloaders import ImageSegDataset
    transforms_list = []
    if config.normstd:
        transforms_list.append(Normalize(mean=0.5, std=0.5))
    transforms_list.append(ToTensor())
    transform_eval = transforms.Compose(transforms_list)

    #dataset validation
    dataset_val = ImageBlindSpotDataset(pd_files_scribbles,transform=transform_eval,validation = True,
                                             ratio=1, size_window=config.size_window,
                                             p_scribble_crop = config.p_scribble_crop, shift_crop = config.shift_crop,
                                             patch_size=config.patch_size, npatch_image = config.npatch_image_sampler)
    dataset_val.sample_patches_data(npatches_total = config.npatches_epoch) #sample first epoch patches
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.BATCH_SIZE,
                                                 shuffle=False, num_workers=config.n_workers)

    #dataloader full images evaluation
    dataloader_eval = torch.utils.data.DataLoader(ImageSegDataset(pd_files,transform = transform_eval),
                                                  batch_size=int(np.minimum((len(pd_files)),16)),
                                                  shuffle=False, num_workers=8)


    #------------------------- Network/Optimizer/Criteria --------------------------------#
    ### Network Unet ###
    from general.networks import UNetFactors

    # n_channels, n_outputs, n_factors, n_classes, bilinear = True, base = 32, depth = 4, activation = 'relu', batchnorm = True)
    model = UNetFactors(config.n_channels,config.n_output,config.nfactors,config.nclasses,
                 depth = config.unet_depth ,
                 base = config.unet_base,
                 activation = config.activation,
                 batchnorm = config.batchnorm)
    model = model.to(config.DEVICE)

    from general.losses import seglosses, reclosses
    criterio_rec = reclosses(type_loss=config.rec_loss, reduction=None)
    criterio_seg = seglosses(type_loss=config.seg_loss, reduction=None)

    from Impartial.Impartial_functions import compute_impartial_losses
    def criterio(out, x, scribble, mask):
        return compute_impartial_losses(out, x, scribble, mask, config, criterio_seg, criterio_rec)

    from torch import optim
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.optim_weight_decay)

    else:
        if config.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE,
                                      weight_decay=config.optim_weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    print('-- Network : ')
    print(model)
    print()

    print('-- Optimizer : ')
    print(optimizer)
    print()

    if cparser.load:
        if os.path.exists(config.basedir + config.model_name + '/' + config.best_model):
            print(' Loading : ',config.basedir + config.model_name + '/' + config.best_model)
            model_params_load(config.basedir + config.model_name + '/' + config.best_model, model, optimizer,config.DEVICE)

    #------------------------- Training --------------------------------#
    from general.training import recseg_trainer
    history = recseg_trainer(dataloader_train,dataloader_val,model,optimizer,criterio,config)

    print(' Saving .... ')
    config.save_json()
    for key in history.keys():
        history[key] = np.array(history[key]).tolist()

    from general.utils import save_json
    save_json(history, config.basedir + config.model_name + '/history.json')
    print('history file saved on : ', config.basedir + config.model_name + '/history.json')

    #------------------------- Evaluation --------------------------------#

    from Impartial.Impartial_functions import get_impartial_outputs
    from general.evaluation import get_performance
    import pickle

    th_list = np.linspace(0,1,21)[1:-1]
    pd_rows = []
    pd_saves_out = []
    ix_file = -1
    for batch, data in enumerate(dataloader_eval):
        print(batch)
        Xinput = data['input'].to(config.DEVICE)

        model.eval()
        out = model(Xinput)
        output = get_impartial_outputs(out, config)

        Ylabels = data['label'].numpy()
        X = data['input'].numpy()

        if cparser.saveout:
            ## Save optional
            save_output_dic = config.basedir + config.model_name + '/output_images/'
            file_output_save = 'eval_batch' + str(batch) + '.pickle'
            mkdir(save_output_dic)
            with open(save_output_dic + file_output_save, 'wb') as handle:
                pickle.dump(output, handle)

        for i in np.arange(X.shape[0]):
            ix_file += 1
            print('Segmentation :')
            pd_saves_out.append([pd_files.iloc[ix_file]['prefix'], file_output_save, batch, i])

            ix_label = 0
            for class_tasks_key in config.classification_tasks.keys():
                classification_tasks = config.classification_tasks[class_tasks_key]
                nclasses = int(classification_tasks['classes'])  # number of classes

                #             Y_back_save.append(np.sum(output[class_tasks_key]['back'][i,:,...],axis = 0))
                for ix_class in range(nclasses):

                    Ypred_fore = np.sum(output[class_tasks_key]['fore_class' + str(ix_class)][i, :, ...], axis=0)
                    Ypred_back = 1 - Ypred_fore  # one vs all evaluation
                    #                 Y_fore_save.append(Ypred_fore)

                    Ylabel = Ylabels[i, int(ix_label), ...].astype('int')
                    ix_label += 1

                    for th in th_list:
                        rows = [_ for _ in pd_files.iloc[ix_file].values]
                        rows.append(class_tasks_key)
                        rows.append(ix_class)
                        rows.append(th)
                        metrics = get_performance(Ylabel, Ypred_fore, threshold=th)
                        for key in metrics.keys():
                            rows.append(metrics[key])
                        pd_rows.append(rows)

    columns = list(pd_files.columns)
    columns.extend(['task', 'segclass', 'th'])
    for key in metrics.keys():
        columns.append(key)
    pd_summary = pd.DataFrame(data=pd_rows, columns=columns)
    pd_summary.to_csv(config.basedir + config.model_name + '/pd_summary_results.csv', index=0)
    print('Evaluation csv saved on : ',
          config.basedir + config.model_name + '/pd_summary_results.csv')

    if cparser.saveout:
        pd_saves = pd.DataFrame(data=pd_saves_out,
                                columns=['prefix', 'output_file', 'batch', 'index'])
        pd_saves.to_csv(save_output_dic + 'pd_output_saves.csv', index=0)
        print('output saves...')