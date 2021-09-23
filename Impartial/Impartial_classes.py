import argparse
import sys
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

sys.path.append("../")
from general.utils import model_params_load, mkdir, save_json, to_np
from Impartial.Impartial_functions import get_impartial_outputs
from general.evaluation import get_performance
from dataprocessing.dataloaders import Normalize, ToTensor, RandomFlip, ImageSegDataset, ImageBlindSpotDataset

class ImPartialConfig(argparse.Namespace):

    def __init__(self, config_dic=None, **kwargs):

        self.basedir = 'models/'  # todo gs
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.last_model = 'weights_last.pth'

        self.seed = 42
        self.GPU_ID = 0

        ### Checkpoint ensembles
        self.nsaves = 1 # number of checkpoints ensembles
        self.val_model_saves_list = []
        self.train_model_saves_list = []
        self.reset_optim = True
        self.reset_validation = False

        ### MC dropout ###
        self.MCdrop = False
        self.MCdrop_it = 40

        ### Network ###
        self.activation = 'relu'
        self.batchnorm = False
        self.unet_depth = 4
        self.unet_base = 32
        self.dropout = False
        self.drop_last_conv = False
        self.drop_encoder_decoder = False
        self.p_drop = 0.5

        ####  Dataloaders  ####
        self.n_channels = 1
        self.patch_size = (128, 128)
        self.p_scribble_crop = 0.6 #probability of sampling a patch with a scribble
        self.shift_crop = 32 #random shift window for the sampled center of the patch
        self.nepochs_sample_patches = 10 #number of epochs until sample patches is available again
        self.npatch_image_sampler = 8 #number of patches to sample when loading an image
        self.npatches_epoch = 512 #number of patches that are considered an epoch

        self.BATCH_SIZE = 32 #batch size
        self.n_workers = 32
        self.augmentations = True
        self.normstd = False #normalization

        #blind spots specific
        self.ratio = 0.95 #(1-ratio)*num_pixels in images will be blind spot
        self.size_window = (10, 10) #window to sample the value of the blind pixel

        ### Losses ###
        self.seg_loss = 'CE'
        self.rec_loss = 'gaussian'
        self.reg_loss = 'L1'
        self.classification_tasks = {'0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [1,2]}}  # list containing classes 'object types'
        self.mean = True
        self.std = False
        self.weight_tasks = None #per segmentation class reconstruction weight
        self.weight_objectives = {'seg_fore':0.45, 'seg_back':0.45, 'rec':0.1, 'reg': 0.0}

        ### training ###
        self.EPOCHS = 100
        self.LEARNING_RATE = 1e-4
        self.lrdecay = 1
        self.optim_weight_decay = 0
        self.patience = 5
        self.optimizer = 'adam'
        self.max_grad_clip = 0 #default
        self.val_stopper = True
        self.type_metric = []
        self.warmup_epochs = 80 #minimum of epochs to consider before allow stopper

        if config_dic is not None:
            self.set_values(config_dic)

        for k in kwargs:
            setattr(self, k, kwargs[k])

        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.GPU_ID >= 0:
            DEVICE = torch.device('cuda:%d' % (self.GPU_ID))
        else:
            DEVICE = torch.device('cpu')
        self.DEVICE = DEVICE

        self.n_output = 0
        for key in self.classification_tasks.keys():
            ncomponents = np.sum(np.array(self.classification_tasks[key]['ncomponents']))
            nrec = len(self.classification_tasks[key]['rec_channels'])
            nclasses = self.classification_tasks[key]['classes']
            self.n_output += ncomponents #components of the task
            if self.mean:
                self.n_output += ncomponents * nrec
            if self.std:
                self.n_output += ncomponents * nrec

            if 'weight_classes' not in self.classification_tasks[key].keys():
                self.classification_tasks[key]['weight_classes'] = [1/nclasses for _ in range(nclasses)]
            if 'weight_rec_channels' not in self.classification_tasks[key].keys():
                self.classification_tasks[key]['weight_rec_channels'] = [1/nrec for _ in range(nrec)]

        if self.weight_tasks is None:
            self.weight_tasks = {}
            for key in self.classification_tasks.keys():
                self.weight_tasks[key] = 1/len(self.classification_tasks.keys())

        for i in range(self.nsaves):
            self.val_model_saves_list.append('model_val_best_save' + str(i) + '.pth')
            self.train_model_saves_list.append('model_train_best_save' + str(i) + '.pth')

        if self.MCdrop & (not (self.drop_last_conv)) & (not (self.drop_encoder_decoder)):
            self.drop_last_conv = False
            self.drop_encoder_decoder = True


    def update_parameters(self, allow_new=True, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])


    def save_json(self,save_path=None):
        config_dict = self.__dict__
        config2json = {}
        for key in config_dict.keys():
            # print(key)
            if key != 'DEVICE':
                if type(config_dict[key]) is np.ndarray:
                    config2json[key] = config_dict[key].tolist()
                elif isinstance(config_dict[key], (np.int_, np.intc, np.intp, np.int8,
                                      np.int16, np.int32, np.int64, np.uint8,
                                      np.uint16, np.uint32, np.uint64)):
                    config2json[key] = int(config_dict[key])
                else:
                    config2json[key] = config_dict[key]
        if save_path is None:
            save_path =  self.basedir + self.model_name + '/config.json'
        save_json(config2json, save_path)
        print('Saving config json file in : ', save_path)

    def set_values(self, config_dic):
        if config_dic is not None:
            for k in config_dic.keys():
                setattr(self, k, config_dic[k])



class ImPartialModel:
    def __init__(self, config):
        self.config = config

        ### Network Unet ###
        from general.networks import UNet
        self.model = UNet(config.n_channels, config.n_output,
                     depth=config.unet_depth,
                     base=config.unet_base,
                     activation=config.activation,
                     batchnorm=config.batchnorm, dropout=self.config.drop_encoder_decoder,
                     dropout_lastconv=self.config.drop_last_conv, p_drop=self.config.p_drop)

        self.model = self.model.to(config.DEVICE)

        from torch import optim
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.optim_weight_decay)

        else:
            if config.optimizer == 'RMSprop':
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=config.LEARNING_RATE,
                                          weight_decay=config.optim_weight_decay)
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE)

        print('---------------- Impartial model config created ----------------------------')
        print()
        print('Model directory:', self.config.basedir + self.config.model_name + '/')
        print()
        print('-- Config file :')
        print(self.config)
        print('')
        print()
        print('-- Network : ')
        print(self.model)
        print()
        print()
        print('-- Optimizer : ')
        print(self.optimizer)
        print()
        print()
        mkdir(self.config.basedir)
        mkdir(self.config.basedir + self.config.model_name + '/')  # todo gs

        self.dataloader_train = None
        self.dataloader_val = None
        self.history = None


    def load_network(self, load_file=None):
        if os.path.exists(load_file):
            print(' Loading : ', load_file)
            model_params_load(load_file, self.model, self.optimizer, self.config.DEVICE)


    def load_dataloaders(self, pd_files_scribbles, pd_files):

        # ------------------------- Dataloaders --------------------------------#
        print('-- Dataloaders : ')
        ### Dataloaders for training
        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        if self.config.augmentations:
            transforms_list.append(RandomFlip())
        transforms_list.append(ToTensor(dim_data=3))

        transform_train = transforms.Compose(transforms_list)

        # dataaset train
        dataset_train = ImageBlindSpotDataset(pd_files_scribbles, transform=transform_train, validation=False,
                                              ratio=self.config.ratio, size_window=self.config.size_window,
                                              p_scribble_crop=self.config.p_scribble_crop, shift_crop=self.config.shift_crop,
                                              patch_size=self.config.patch_size, npatch_image=self.config.npatch_image_sampler)
        
        print('Sampling ' + str(self.config.npatches_epoch) + ' train patches ... ')

        dataset_train.sample_patches_data(npatches_total=self.config.npatches_epoch)  # sample first epoch patches
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.config.BATCH_SIZE, shuffle=True,
                                                       num_workers=self.config.n_workers)


        ### Dataloader for validation
        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        transforms_list.append(ToTensor())
        transform_val = transforms.Compose(transforms_list)

        # dataset validation
        dataset_val = ImageBlindSpotDataset(pd_files_scribbles, transform=transform_val, validation=True,
                                            ratio=1, size_window=self.config.size_window,
                                            p_scribble_crop=self.config.p_scribble_crop, shift_crop=self.config.shift_crop,
                                            patch_size=self.config.patch_size, npatch_image=self.config.npatch_image_sampler)
        
        print('Sampling ' + str(self.config.npatches_epoch) + ' validation patches ...')
        dataset_val.sample_patches_data(npatches_total=self.config.npatches_epoch)  # sample first epoch patches
        self.dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.config.BATCH_SIZE,
                                                     shuffle=False, num_workers=self.config.n_workers)


        # dataloader full images evaluation
        # batch size 1 !!!      #gs
        batch_size = 1
        dataset_eval = ImageSegDataset(pd_files, transform=transform_val)
        self.dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=8) 



    def train(self):

        if (self.dataloader_train == None) or (self.dataloader_val == None) or (self.dataloader_eval == None):
            print('No train/val/eval dataloader was loaded')
            return

        # ------------------------- losses --------------------------------#
        from general.losses import seglosses, reclosses, gradientLoss2d

        criterio_seg = seglosses(type_loss=self.config.seg_loss, reduction=None)
        criterio_reg = None
        if 'reg' in self.config.weight_objectives.keys():
            if self.config.weight_objectives['reg'] > 0:
                criterio_reg = gradientLoss2d(penalty=self.config.reg_loss, reduction='mean')

        criterio_rec = None
        if 'rec' in self.config.weight_objectives.keys():
            if self.config.weight_objectives['rec'] > 0:
                criterio_rec = reclosses(type_loss=self.config.rec_loss, reduction=None)

        from Impartial.Impartial_functions import compute_impartial_losses
        def criterio(out, x, scribble, mask):
            return compute_impartial_losses(out, x, scribble, mask, self.config,
                                            criterio_seg, criterio_rec, criterio_reg=criterio_reg)



        # ------------------------- Training --------------------------------#
        from general.training import recseg_checkpoint_ensemble_trainer
        history = recseg_checkpoint_ensemble_trainer(self.dataloader_train, self.dataloader_val, self.dataloader_eval,
                                                     self.model, self.optimizer, criterio, self.config)

        for key in history.keys():
            history[key] = np.array(history[key]).tolist()

        from general.utils import save_json
        save_json(history, self.config.basedir + self.config.model_name + '/history.json')    # todo gs
        print('history file saved on: ', self.config.basedir + self.config.model_name + '/history.json')   # todo gs

        return history
    

    def data_performance_evaluation(self, pd_files, saveout=False, plot=False, default_ensembles=True, model_ensemble_load_files=[]):
        
        start_eval_time = time.time() 
        # ------------ Dataloader ----------#
        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        transforms_list.append(ToTensor())
        transform_eval = transforms.Compose(transforms_list)

        # dataloader full images evaluation
        batch_size = 1
        dataloader_eval = torch.utils.data.DataLoader(ImageSegDataset(pd_files, transform=transform_eval),
                                                      batch_size=batch_size, shuffle=False, num_workers=8) ## Batch size 1 !!!

        
        # ------------------------- Evaluation --------------------------------#
        from general.training import eval
        output_list, gt_list = eval(dataloader_eval, self.model, self.optimizer, self.config, epoch=0, saveout=False, 
                                    default_ensembles=default_ensembles, model_ensemble_load_files=model_ensemble_load_files)
        
        end_eval_time = time.time() #gs
        print('Evaluation time taken:  ', str(end_eval_time - start_eval_time))

        th_list = np.linspace(0, 1, 21)[1:-1]
        pd_rows = []
        pd_saves_out = []

        for ix_file in range(len(pd_files)):
            output = output_list[ix_file]
            Ylabels = gt_list[ix_file]

            print('Performance evaluation on file ', pd_files.iloc[ix_file]['prefix'])
            for task in self.config.classification_tasks.keys():

                output_task = output[task]

                ix_labels_list = self.config.classification_tasks[task]['ix_gt_labels']

                for ix_class in range(self.config.classification_tasks[task]['classes']):
                    print(' task : ', task, 'class : ', ix_class)
                    ix_labels = int(ix_labels_list[ix_class])

                    Ypred_fore = output_task['class_segmentation'][0, int(ix_class), ...]
                    Ylabel = Ylabels[0, ix_labels, ...].astype('int')

                    if plot:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(Ylabel)
                        plt.subplot(1, 2, 2)
                        plt.imshow(Ypred_fore)
                        plt.colorbar()
                        plt.show()

                    ix_labels += 1
                    for th in th_list:
                        rows = [_ for _ in pd_files.iloc[ix_file].values]
                        rows.append(task)
                        rows.append(ix_class)
                        rows.append(th)
                        metrics = get_performance(Ylabel, Ypred_fore, threshold=th)
                        for key in metrics.keys():
                            rows.append(metrics[key])
                        pd_rows.append(rows)

            ## Save output (optional)
            if saveout:
                prefix = pd_files.iloc[ix_file]['prefix']
                save_output_dic = self.config.basedir + self.config.model_name + '/output_images/'
                file_output_save = 'eval_' + prefix + '.pickle'
                mkdir(save_output_dic)
                print('Saving output : ', save_output_dic + file_output_save)
                with open(save_output_dic + file_output_save, 'wb') as handle:
                    pickle.dump(output, handle)
                pd_saves_out.append([prefix, file_output_save])

        columns = list(pd_files.columns)
        columns.extend(['task', 'segclass', 'th'])
        for key in metrics.keys():
            columns.append(key)
        pd_summary = pd.DataFrame(data=pd_rows, columns=columns)

        if saveout:
            pd_saves = pd.DataFrame(data=pd_saves_out, columns=['prefix', 'output_file'])
            pd_saves.to_csv(save_output_dic + 'pd_output_saves.csv', index=0)
            print('pandas outputs file saved in :', save_output_dic + 'pd_output_saves.csv')

        return pd_summary

