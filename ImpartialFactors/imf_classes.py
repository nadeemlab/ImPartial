
import argparse
import sys
sys.path.append("../")
from general.utils import save_json
import torch
import numpy as np

class ImPartialFactorsConfig(argparse.Namespace):

    def __init__(self,**kwargs):

        self.basedir = 'models/'
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.last_model = 'weights_last.pth'

        self.seed = 42
        self.GPU_ID = 0

        #Network
        self.activation = 'relu'
        self.batchnorm = False
        self.unet_depth = 4
        self.unet_base = 32


        ####  Dataloaders  ####
        self.n_channels = 1
        self.patch_size = (128, 128)
        self.p_scribble_crop = 0.6 #probability of sampling a patch with a scribble
        self.shift_crop = 32 # random shift window for the sampled center of the patch
        self.nepochs_sample_patches = 4 #number of epochs until resample patches
        self.npatch_image_sampler = 8 #number of patches to sample when loading an image
        self.npatches_epoch = 512 #number of patches that are considered an epoch

        self.BATCH_SIZE = 32 #batch size
        self.n_workers = 32
        self.augmentations = True
        self.normstd = False #normalization

        #blind spots specific
        self.ratio = 0.95 #(1-ratio)*num_pixels in images will be blind spot
        self.size_window = (10,10) #window to sample the value of the blind pixel

        ### Losses ###
        self.seg_loss = 'CE'
        self.rec_loss = 'gaussian'
        # self.segclasses_channels = {'0': [0]}  #list containing classes 'object types'
        self.nclasses = 1  # list containing classes 'object types'
        self.nfactors = self.nclasses + 2 #number of classes plus background and one residual is default option
        self.mean = True
        self.std = False
        self.weight_classes = None #per segmentation class reconstruction weight
        self.weight_rec_channels = None  # per segmentation class reconstruction weight
        self.weight_objectives = {'seg_fore':0.25,'seg_back':0.25,'rec':0.5}


        self.EPOCHS = 100
        self.LEARNING_RATE = 1e-4
        self.lrdecay = 1
        self.optim_weight_decay = 0
        self.patience = 10
        self.optimizer = 'adam'
        self.max_grad_clip = 0 #default no clipping
        self.val_stopper = True
        self.type_metric = []
        self.warmup_epochs = 80 #minimum of epochs to consider before allow stopper

        for k in kwargs:
            setattr(self, k, kwargs[k])

        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.GPU_ID >= 0:
            DEVICE = torch.device('cuda:%d' % (self.GPU_ID))
        else:
            DEVICE = torch.device('cpu')
        self.DEVICE = DEVICE

        self.n_output = self.nfactors
        if self.mean:
            self.n_output += self.nfactors*self.n_channels
        if self.std:
            self.n_output += self.nfactors*self.n_channels

        if self.weight_classes is None:
            self.weight_classes = [1/self.nclasses for _ in range(self.nclasses)]

        if self.weight_rec_channels is None:
            self.weight_rec_channels = [1/self.n_channels for _ in range(self.n_channels)]

    # def is_valid(self, return_invalid=False):
        # Todo! I have to update this properly
        # ok = {}
        #
        # if return_invalid:
        #     return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        # else:
        #     return all(ok.values())

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