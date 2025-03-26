import argparse
import os
import numpy as np
import torch

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

        self.save_intermediates = False

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
        # self.patch_size = (128, 128)
        self.patch_size = (256, 256)
        self.p_scribble_crop = 0.6 #probability of sampling a patch with a scribble
        self.shift_crop = 32 #random shift window for the sampled center of the patch
        self.nepochs_sample_patches = 10 #number of epochs until sample patches is available again
        self.npatch_image_sampler = 8 #number of patches to sample when loading an image
        self.npatches_epoch = 512 #number of patches that are considered an epoch

        self.BATCH_SIZE = 32 #batch size
        self.n_workers = 24
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

        def get_output_size():
            output_size = 0
            for task in self.classification_tasks.values():
                ncomponents = np.sum(np.array(task['ncomponents']))
                output_size += ncomponents #components of the task

                nrec = len(task['rec_channels'])
                output_size += ncomponents * nrec

            return output_size

        self.n_output = get_output_size()

        for task in self.classification_tasks.values():
            nclasses = task['classes']
            nrec = len(task['rec_channels'])

            if 'weight_classes' not in task.keys():
                task['weight_classes'] = [1.0/nclasses for _ in range(nclasses)]
            if 'weight_rec_channels' not in task.keys():
                task['weight_rec_channels'] = [1.0/nrec for _ in range(nrec)]

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


    def set_values(self, config_dic):
        if config_dic is not None:
            for k in config_dic.keys():
                setattr(self, k, config_dic[k])
