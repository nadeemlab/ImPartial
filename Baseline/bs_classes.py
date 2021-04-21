
import argparse
import sys
sys.path.append("../")
from general.utils import save_json,model_params_load,mkdir
import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

class Config(argparse.Namespace):

    def __init__(self,config_dic = None,**kwargs):

        self.basedir = 'models/'
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.last_model = 'weights_last.pth'

        self.seed = 42
        self.GPU_ID = 0

        ### Checkpoint ensembles
        self.nsaves = 1 #number of checkpoints ensembles
        self.val_model_saves_list = []
        self.train_model_saves_list = []
        self.reset_optim = True

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

        # self.segclasses = ['0']  #list containing classes 'object types'
        self.classification_tasks = {'0': {'classes': 1, 'ncomponents': [1, 2]}}
        self.weight_tasks = None
        self.weight_objectives = {'seg_fore':0.5,'seg_back':0.5}

        self.EPOCHS = 100
        self.LEARNING_RATE = 1e-4
        self.lrdecay = 1
        self.optim_weight_decay = 0
        self.patience = 10
        self.optimizer = 'adam'
        self.max_grad_clip = 0  # default no clipping
        self.val_stopper = True
        self.type_metric = []
        self.warmup_epochs = 80 #minimum of epochs to consider before allow stopper

        if config_dic is not None:
            self.set_values(config_dic)
            # for k in config_dic.keys():
            #     setattr(self, k, config_dic[k])

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
            nclasses = self.classification_tasks[key]['classes']
            self.n_output += ncomponents #components of the task
            if 'weight_classes' not in self.classification_tasks[key].keys():
                self.classification_tasks[key]['weight_classes'] = [1/nclasses for _ in range(nclasses)]

        if self.weight_tasks is None:
            self.weight_tasks = {}
            for key in self.classification_tasks.keys():
                self.weight_tasks[key] = 1 / len(self.classification_tasks.keys())

        for i in range(self.nsaves):
            self.val_model_saves_list.append('model_val_best_save' + str(i) + '.pth')
            self.train_model_saves_list.append('model_train_best_save' + str(i) + '.pth')


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

    def set_values(self,config_dic):
        if config_dic is not None:
            for k in config_dic.keys():
                setattr(self, k, config_dic[k])


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



class BaselineModel:
    def __init__(self, config):
        self.config = config

        ### Network Unet ###
        from general.networks import UNet
        self.model = UNet(config.n_channels, config.n_output,
                     depth=config.unet_depth,
                     base=config.unet_base,
                     activation=config.activation,
                     batchnorm=config.batchnorm)

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


    def load_network(self, load_file=None):
        import os
        if os.path.exists(load_file):
            print(' Loading : ', load_file)
            model_params_load(load_file, self.model, self.optimizer,self.config.DEVICE)

    def data_performance_evaluation(self,pd_files,saveout = False, plot = False, model_ensemble_load_files = []):
        from Baseline.bs_functions import get_outputs
        from general.evaluation import get_performance
        from dataprocessing.dataloaders import Normalize, ToTensor, ImageSegDataset
        from torchvision import transforms

        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        transforms_list.append(ToTensor())
        transform_eval = transforms.Compose(transforms_list)

        # dataloader full images evaluation
        dataloader_eval = torch.utils.data.DataLoader(ImageSegDataset(pd_files, transform=transform_eval),
                                                      batch_size=int(np.minimum((len(pd_files)), 16)),
                                                      shuffle=False, num_workers=8)

        th_list = np.linspace(0, 1, 21)[1:-1]
        pd_rows = []
        pd_saves_out = []
        ix_file = -1
        for batch, data in enumerate(dataloader_eval):
            # print(batch)
            Xinput = data['input'].to(self.config.DEVICE)

            if len(model_ensemble_load_files) < 1:
                self.model.eval()
                out = self.model(Xinput)
                output = get_outputs(out, self.config)  # output has keys: class_segmentation, factors
            else:
                ix_model = 0
                for model_save in model_ensemble_load_files:
                    print('evaluation of model : ',model_save)
                    model_params_load(model_ensemble_load_files, self.model, self.optimizer, self.config.DEVICE)
                    self.model.eval()
                    out = self.model(Xinput)
                    output_aux = get_outputs(out, self.config)
                    ix_model += 1
                    if ix_model == 1:  # first model
                        output = output_aux.copy()
                    else:
                        for task in self.config.classification_tasks.keys():
                            for ix_class in range(self.config.classification_tasks[task]['classes']):
                                output[task]['class_segmentation'] = (output_aux[task]['class_segmentation'] +
                                                                      output[task]['class_segmentation'] * (
                                                                      ix_model - 1)) / ix_model

            Ylabels = data['label'].numpy()
            X = data['input'].numpy()

            if saveout:
                ## Save optional
                save_output_dic = self.config.basedir + self.config.model_name + '/output_images/'
                file_output_save = 'eval_batch' + str(batch) + '.pickle'
                mkdir(save_output_dic)
                with open(save_output_dic + file_output_save, 'wb') as handle:
                    pickle.dump(output, handle)

            for i in np.arange(X.shape[0]):
                ix_file += 1
                print('Evaluation on sample :', i)
                if saveout:
                    pd_saves_out.append([pd_files.iloc[ix_file]['prefix'], file_output_save, batch, i])

                ix_labels = 0
                for task in self.config.classification_tasks.keys():
                    output_task = output[task]
                    for ix_class in range(self.config.classification_tasks[task]['classes']):

                        Ypred_fore = output_task['class_segmentation'][i, int(ix_class), ...]
                        Ylabel = Ylabels[i, ix_labels, ...].astype('int')

                        if plot:
                            plt.figure(figsize=(10, 5))
                            plt.subplot(1, 2, 1)
                            plt.imshow(Ylabel)
                            plt.subplot(1, 2, 2)
                            plt.imshow(Ypred_fore)
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

        columns = list(pd_files.columns)
        columns.extend(['task', 'segclass', 'th'])
        for key in metrics.keys():
            columns.append(key)
        pd_summary = pd.DataFrame(data=pd_rows, columns=columns)

        if saveout:
            pd_saves = pd.DataFrame(data=pd_saves_out,
                                    columns=['prefix', 'output_file', 'batch', 'index'])
            pd_saves.to_csv(save_output_dic + 'pd_output_saves.csv', index=0)
            print('output saved in :',save_output_dic + 'pd_output_saves.csv')

        return pd_summary


