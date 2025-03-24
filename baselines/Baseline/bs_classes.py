
import argparse
import sys
import os
import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append("../")

from general.utils import save_json,model_params_load,mkdir,to_np
from Baseline.bs_functions import get_outputs
from dataprocessing.dataloaders import Normalize, ToTensor, RandomFlip, ImageSegDataset,ImageBlindSpotDataset
from general.evaluation import get_performance



class Config(argparse.Namespace):

    def __init__(self, config_dic=None, **kwargs):

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
        self.reset_validation = False

        ## MC dropout
        self.MCdrop = False
        self.MCdrop_it = 40


        #Network
        self.activation = 'relu'
        self.batchnorm = False
        self.unet_depth = 4
        self.unet_base = 32
        self.drop_last_conv = False
        self.drop_encoder_decoder = False
        self.p_drop = 0.5


        ####  Dataloaders  ####
        self.n_channels = 1
        self.patch_size = (128, 128)
        self.p_scribble_crop = 0.6 #probability of sampling a patch with a scribble
        self.shift_crop = 32 # random shift window for the sampled center of the patch
        self.nepochs_sample_patches = 10 #number of epochs until resample patches
        self.npatch_image_sampler = 8 #number of patches to sample when loading an image
        self.npatches_epoch = 4096 #number of patches that are considered an epoch

        self.BATCH_SIZE = 64 #batch size
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
        self.LEARNING_RATE = 5e-4
        self.lrdecay = 1
        self.optim_weight_decay = 0
        self.patience = 5
        self.optimizer = 'adam'
        self.max_grad_clip = 0  # default no clipping
        self.val_stopper = True
        self.type_metric = []
        self.warmup_epochs = 50 #minimum of epochs to consider before allow stopper

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

        if self.MCdrop & (not(self.drop_last_conv)) & (not(self.drop_encoder_decoder)):
            self.drop_last_conv = False
            self.drop_encoder_decoder = True

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

    def set_values(self, config_dic):
        if config_dic is not None:
            for k in config_dic.keys():
                setattr(self, k, config_dic[k])


    def save_json(self, save_path=None):
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
            # save_path =  self.basedir + self.model_name + '/config.json'
            save_path = os.path.join(self.basedir, self.model_name, 'config.json')
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

        self.model_output_dir = os.path.join(self.config.basedir, self.config.model_name)
        print('---------------- Baseline model config created ----------------------------')
        print()
        print('Model directory:', self.model_output_dir)
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
        mkdir(self.model_output_dir)

        self.dataloader_train = None
        self.dataloader_val = None
        self.history = None

    def load_network(self, load_file=None):
        import os
        if os.path.exists(load_file):
            print(' Loading : ', load_file)
            model_params_load(load_file, self.model, self.optimizer,self.config.DEVICE)

    def load_dataloaders(self, data_dir, pd_files_scribbles):

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
        dataset_train = ImageBlindSpotDataset(pd_files_scribbles, data_dir, transform=transform_train, validation=False,
                                              ratio=self.config.ratio, size_window=self.config.size_window,
                                              p_scribble_crop=self.config.p_scribble_crop, shift_crop=self.config.shift_crop,
                                              patch_size=self.config.patch_size, npatch_image=self.config.npatch_image_sampler)
        print('Sampling ' + str(self.config.npatches_epoch) + ' train patches ... ')
        dataset_train.sample_patches_data(npatches_total=self.config.npatches_epoch)  # sample first epoch patches
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.config.BATCH_SIZE, shuffle=True,
                                                       num_workers=self.config.n_workers)

        ### Dataloader for evaluation
        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        transforms_list.append(ToTensor())
        transform_eval = transforms.Compose(transforms_list)

        # dataset validation
        dataset_val = ImageBlindSpotDataset(pd_files_scribbles, data_dir, transform=transform_eval, validation=True,
                                            ratio=1, size_window=self.config.size_window,
                                            p_scribble_crop=self.config.p_scribble_crop, shift_crop=self.config.shift_crop,
                                            patch_size=self.config.patch_size, npatch_image=self.config.npatch_image_sampler)
        print('Sampling ' + str(self.config.npatches_epoch) + ' validation patches ... ')
        dataset_val.sample_patches_data(npatches_total=self.config.npatches_epoch)  # sample first epoch patches
        self.dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.config.BATCH_SIZE,
                                                     shuffle=False, num_workers=self.config.n_workers)

    def eval(self, pd_files, data_dir, default_ensembles=True, model_ensemble_load_files=[]):

        if default_ensembles & (len(model_ensemble_load_files) < 1):
            model_ensemble_load_files = []
            for model_file in self.config.val_model_saves_list:
                model_ensemble_load_files.append(os.path.join(self.model_output_dir, model_file))

        if len(model_ensemble_load_files) >= 1:
            print('Evaluating average predictions of models : ')
            for model_file in model_ensemble_load_files:
                print(model_file)
        elif not default_ensembles:
            print('Evaluation of currently loaded network')

        # ------------ Dataloader ----------#
        transforms_list = []
        if self.config.normstd:
            transforms_list.append(Normalize(mean=0.5, std=0.5))
        transforms_list.append(ToTensor())
        transform_eval = transforms.Compose(transforms_list)

        # dataloader full images evaluation
        batch_size = 1
        dataloader_eval = torch.utils.data.DataLoader(ImageSegDataset(pd_files, data_dir, transform=transform_eval),
                                                      batch_size=batch_size, shuffle=False, num_workers=8) ## Batch size 1 !!!

        # ---------- Evaluation --------------#
        output_list = []
        gt_list = []
        for batch, data in enumerate(dataloader_eval):
            # print(batch)
            Xinput = data['input'].to(self.config.DEVICE)

            ## save ground truth
            if 'label' in data.keys():
                Ylabel = data['label'].numpy()
                gt_list.append(Ylabel)

            ## evaluate ensemble of checkpoints and save outputs
            if len(model_ensemble_load_files) < 1:
                self.model.eval()
                with torch.no_grad():
                    predictions = (self.model(Xinput)).cpu().numpy()
            else:
                predictions = np.empty((0, batch_size, self.config.n_output, Xinput.shape[-2], Xinput.shape[-1]))
                for model_save in model_ensemble_load_files:
                    if os.path.exists(model_save):
                        # print('evaluation of model : ',model_save)
                        model_params_load(model_save, self.model, self.optimizer, self.config.DEVICE)
                        self.model.eval()

                        if self.config.MCdrop:
                            self.model.enable_dropout()
                            for it in range(self.config.MCdrop_it):
                                # print(it)
                                with torch.no_grad():
                                    out = to_np(self.model(Xinput))
                                predictions = np.vstack((predictions, out[np.newaxis,...]))
                        else:
                            with torch.no_grad():
                                out = to_np(self.model(Xinput))
                            predictions = np.vstack((predictions, out[np.newaxis, ...]))

            output = get_outputs(predictions, self.config)  # output has keys: class_segmentation, factors
            output_list.append(output)
        return output_list, gt_list

    def train(self):

        if (self.dataloader_train is not None) & (self.dataloader_val is not None):

            # ------------------------- losses -------------------------------- #
            from general.losses import seglosses
            criterio_seg = seglosses(type_loss=self.config.seg_loss, reduction=None)

            from Baseline.bs_functions import compute_segloss
            def criterio(out, x, scribble, mask):
                return compute_segloss(out, scribble, self.config, criterio_seg)


            # ------------------------- Training -------------------------------- #
            from general.training import recseg_checkpoint_ensemble_trainer
            history = recseg_checkpoint_ensemble_trainer(self.dataloader_train, self.dataloader_val, None, self.model,
                                                         self.optimizer, criterio, self.config)

            for key in history.keys():
                history[key] = np.array(history[key]).tolist()

            from general.utils import save_json
            _path = os.path.join(self.model_output_dir, 'history.json')
            save_json(history, _path)
            self.config.save_json()
            print('history file saved on : ', _path)
            self.history = history
            return history
        else:
            print('No train/val dataloader was loaded')

    def data_performance_evaluation(self, pd_files, data_dir, saveout=False, plot=False, default_ensembles=True,
                                    model_ensemble_load_files=[]):

        output_list, gt_list = self.eval(pd_files, data_dir, default_ensembles=default_ensembles,
                                         model_ensemble_load_files=model_ensemble_load_files)

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
                # save_output_dic = self.config.basedir + self.config.model_name + '/output_images/'
                save_output_dic = os.path.join(self.model_output_dir, 'output_images')
                file_output_save = 'eval_' + prefix + '.pickle'
                mkdir(save_output_dic)
                _path = os.path.join(save_output_dic, file_output_save)
                print('Saving output : ', _path)
                with open(_path, 'wb') as handle:
                    pickle.dump(output, handle)
                pd_saves_out.append([prefix, file_output_save])

        columns = list(pd_files.columns)
        columns.extend(['task', 'segclass', 'th'])
        for key in metrics.keys():
            columns.append(key)
        pd_summary = pd.DataFrame(data=pd_rows, columns=columns)

        if saveout:
            pd_saves = pd.DataFrame(data=pd_saves_out, columns=['prefix', 'output_file'])
            _path = os.path.join(save_output_dic, 'pd_output_saves.csv')    # todo gs
            pd_saves.to_csv(_path, index=0)
            print('pandas outputs file saved in :', _path)

        return pd_summary
