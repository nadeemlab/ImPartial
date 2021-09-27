import sys
import os
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim

sys.path.append("../")
from general.utils import model_params_load, mkdir, to_np, TravellingMean
from Impartial.Impartial_functions import get_impartial_outputs

def epoch_recseg_loss(dataloader, model, optimizer, config, criterio, train_type=True):
    #criterio has to output loss, seg_fore loss, seg_back loss and rec loss

    if train_type:
        model.train()
    else:
        model.eval()
        if config.MCdrop:
            model.enable_dropout()

    output = {}
    output['loss'] = TravellingMean()
    for key in config.weight_objectives.keys():
        output[key] = TravellingMean()

    output['grad'] = TravellingMean()

    # tic_list = []
    # tic_list.append(time.perf_counter())
    # tic_bs_list = []
    # tic_criterio_list = []
    # tic_final_list = []

    for batch, data in enumerate(dataloader):

        x = data['input'].to(config.DEVICE) #input image with blind spots replaced randomly
        mask = data['mask'].to(config.DEVICE)
        scribble = data['scribble'].to(config.DEVICE)
        target = data['target'].to(config.DEVICE) #input image with non blind spots

        # tic_list.append(time.perf_counter())
        # tic_bs_list.append(tic_list[-1] - tic_list[-2])
        # print(' batch sampling : ', tic_list[-1] - tic_list[-2])

        if train_type:
            out = model(x)
            losses = criterio(out, target, scribble, mask)
        else:
            with torch.no_grad():
                out = model(x)
                if config.MCdrop:
                    out = torch.unsqueeze(out, 0)
                    for it in range(2): #loss computed using 2 dropout predictions in total
                        out_aux = model(x)
                        out = torch.cat((out, torch.unsqueeze(out_aux, 0)), 0)

                losses = criterio(out, target, scribble, mask)


        # tic_list.append(time.perf_counter())
        # tic_criterio_list.append(tic_list[-1] - tic_list[-2])
        # print(' criterio : ', tic_list[-1] - tic_list[-2])

        loss_batch = 0
        for key in config.weight_objectives.keys():
            loss_batch += losses[key] * config.weight_objectives[key]
            output[key].update(to_np(losses[key]), mass=x.shape[0])

        output['loss'].update(to_np(loss_batch), mass=x.shape[0])

        if train_type:
            optimizer.zero_grad()
            loss_batch.backward()
            if config.max_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_clip, norm_type=2)
            # else:
                # print('no clipping')

            # total_norm = 0
            # for p in model.parameters():
                # param_norm = p.grad.data.norm(2)
                # total_norm += param_norm.item() ** 2
            # output['grad'].update((total_norm ** (1. / 2)), mass=1)

            optimizer.step()

        # tic_list.append(time.perf_counter())
        # tic_final_list.append(tic_list[-1] - tic_list[-2])
        # print(' losses etc : ', tic_list[-1] - tic_list[-2])

    # print('Total time (batch, criterio, backward, total) : ')
    # print(np.round(np.mean(np.array(tic_bs_list)),3),
          # np.round(np.mean(np.array(tic_criterio_list)),3),
          # np.round(np.mean(np.array(tic_final_list)), 3),np.round(tic_list[-1]-tic_list[0],1))
                    # break

    # print(total_loss)
    for key in output.keys():
        output[key] = output[key].mean
    return output


def recseg_checkpoint_ensemble_trainer(dataloader_train, dataloader_val, dataloader_eval,
                                       model, optimizer, criterio, config):

    from general.training import epoch_recseg_loss
    from general.utils import early_stopping, model_params_save, model_params_load

    ## Savings
    history = {}
    history['cycle'] = []
    history['loss_mbatch_train'] = []
    for key in config.weight_objectives.keys():
        history[key+'_mbatch_train'] = []
    history['grad_mbatch_train'] = []

    for _ in ['val']:
        history['loss_' + _] = []
        for key in config.weight_objectives.keys():
            history[key + '_' + _] = []

    #tags that will be printed
    tags_plot = ['loss']
    for key in config.weight_objectives.keys():
        tags_plot.append(key)


    #number of cycles
    nsaves = config.nsaves
    #boolean: reset optimizer when starting a new cycle
    reset_optim = config.reset_optim

    ## Stoppers and saving lists
    stopper_best_all = early_stopping(config.patience, 0, np.infty) #only to save global best model
    stopper = early_stopping(config.patience*2, 0, np.infty) #first cycle has patience x 2

    if config.val_stopper:
        model_saves_list = config.val_model_saves_list #stores the epochs of each val
    else:
        model_saves_list = config.train_model_saves_list

    epochs_saves_list = [0] #stores the epochs of each saving
    loss_saves_list = [0] #stores the corresponding loss

    patch_epoch_counter = 0 #counter of epochs iteration for the current patches
    patch_sampler = 0 #counter of patches iterations

    if (len(model_saves_list) < nsaves):
        print('!! ERROR : not enough saving files where defined')
        return

    # tic_list = []
    # tic_list.append(time.perf_counter())
    # tic_bs_list = []
    # tic_criterio_list = []
    # tic_final_list = []

    for epoch in range(config.EPOCHS): #config.epochs should be set super large is not supposed to be the stopping criteria

        history['cycle'].append(patch_sampler)
       
        #Training:
        start_train_time = time.time() #gs
        output = epoch_recseg_loss(dataloader_train, model, optimizer, config, criterio, train_type=True)
        end_train_time = time.time() #gs
        
        for key in output.keys():
            history[key + '_mbatch_train'].append(output[key])

        # tic_list.append(time.perf_counter())
        # print( 'epoch train : ',tic_list[-1] - tic_list[-2])

        # Validation:
        start_val_time = time.time()
        output = epoch_recseg_loss(dataloader_val, model, optimizer, config, criterio, train_type=False)
        end_val_time = time.time()

        for key in output.keys():
            if key != 'grad':
                history[key + '_val'].append(output[key])


        if (epoch + 1) % 5 == 0:
            eval(dataloader_eval, model, optimizer, config, epoch)

        # tic_list.append(time.perf_counter())
        # print('epoch val : ', tic_list[-1] - tic_list[-2])


        if config.val_stopper:
            save_best_all, _ = stopper_best_all.evaluate(history['loss_val'][-1])
            save, stop = stopper.evaluate(history['loss_val'][-1])
        else:
            save_best_all, _ = stopper_best_all.evaluate(history['loss_mbatch_train'][-1])
            save, stop = stopper.evaluate(history['loss_mbatch_train'][-1])

        # tic_list.append(time.perf_counter())
        # print( 'epoch stopper : ',tic_list[-1] - tic_list[-2])

        if save_best_all: #global best val model
            history['epoch_best_of_all'] = epoch
            best_model_of_all = int(patch_sampler + 0)
            # model_params_save(config.basedir + config.model_name + '/' + config.best_model, model, optimizer)  # save best model

        if save:
            # print(model_saves_list,patch_sampler)
            # print(model_saves_list[patch_sampler])
            model_params_save(config.basedir + config.model_name + '/' + model_saves_list[patch_sampler], model, optimizer)  # save best model
            epochs_saves_list[-1] = epoch
            loss_saves_list[-1] = stopper.best_loss
            print('saving best model, epoch: ', epoch, ' to : ', model_saves_list[patch_sampler])

        # tic_list.append(time.perf_counter())
        # print( 'epoch saves : ',tic_list[-1] - tic_list[-2])

        string_print = 'epoch : ' + str(epoch) + ' cycle : ' + str(patch_sampler)
        for key in tags_plot:
            string_print = string_print + ' | ' + key + ' tr, val : ' + str(
                np.round(history[key + '_mbatch_train'][-1], 3)) + ','
            string_print = string_print + str(np.round(history[key + '_val'][-1], 3))
        string_print = string_print + ' | stopper_cycle : ' + str(stopper.counter) + '; stopper_best :' + str(stopper_best_all.counter)
        
        string_print = string_print + ' | train_time_taken : ' + str(end_train_time - start_train_time) + ' | val_time_taken : ' + str(end_val_time - start_val_time) # gs 
        
        print(string_print)
        # print(val_epochs_saves_list)
        print('loss of validation checkpoints : ',loss_saves_list)
        print()

        patch_epoch_counter += 1
        if (stop & (patch_epoch_counter>=config.nepochs_sample_patches)) : #if patience criteria and we have trained for a minimum of epochs per cycle of patches

            if (patch_sampler+1 >= nsaves) :
                break
                # if (epoch > config.warmup_epochs):
                #     break
                #if still in warmup epochs we continue training and resample
            # else:

            patch_sampler += 1  # counter of patches iterations
            patch_epoch_counter = 0  # counter of epochs iteration for the current patches

            print('sampling ',str(config.npatches_epoch), ' new training patches...')
            dataloader_train.dataset.sample_patches_data(npatches_total=config.npatches_epoch)
            if config.reset_validation:
                print('sampling ', str(config.npatches_epoch), ' new validation patches...')
                dataloader_val.dataset.sample_patches_data(npatches_total=config.npatches_epoch)

            #reset train stopper and increase train save, note that we save once per sample patches iteration
            stopper.best_loss = np.infty
            stopper.counter = 0
            stopper.patience = config.patience

            # val_model_saves_list.append('model_val_best_save' + str(patch_sampler) + '.pth')
            epochs_saves_list.append(0)
            loss_saves_list.append(0)

            ## reset optimizer?
            if reset_optim:
                print('reset optimizer')
                if config.optimizer == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                           weight_decay=config.optim_weight_decay)
                else:
                    if config.optimizer == 'RMSprop':
                        optimizer = optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE,
                                                  weight_decay=config.optim_weight_decay)
                    else:
                        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

        # tic_list.append(time.perf_counter())
        # print('epoch end : ', tic_list[-1] - tic_list[-2])

    history['val_epochs_saves_list'] = epochs_saves_list
    history['val_loss_saves_list'] = loss_saves_list

    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + model_saves_list[best_model_of_all])
    model_params_load(config.basedir + config.model_name + '/' + model_saves_list[best_model_of_all], model, optimizer, config.DEVICE)
    model_params_save(config.basedir + config.model_name + '/' + config.best_model, model, optimizer)  # save best model

    return history



def eval(dataloader_eval, model, optimizer, config, epoch, saveout=True, default_ensembles=True, model_ensemble_load_files=[]):

    if default_ensembles & (len(model_ensemble_load_files) < 1):
        model_ensemble_load_files = []
        for model_file in config.val_model_saves_list:
            model_ensemble_load_files.append(
                config.basedir + config.model_name + '/' + model_file)

    if len(model_ensemble_load_files) >= 1:
        print('Evaluating average predictions of models : ')
        print("model_ensemble_load_files: ", ' '.join(model_ensemble_load_files))
    elif not default_ensembles:
        print('Evaluation of currently loaded network')
    
    
    # dataloader full images evaluation
    batch_size = 1

    # ---------- Evaluation --------------#
    output_list = []
    gt_list = []
    idx = 0
    print('Start evaluation in training ...')
    for batch, data in enumerate(dataloader_eval):
        print()
        print('batch : ', batch)
        Xinput = data['input'].to(config.DEVICE)

        ## save ground truth
        if 'label' in data.keys():
            Ylabel = data['label'].numpy()
            gt_list.append(Ylabel)

        ## evaluate ensemble of checkpoints and save outputs
        if len(model_ensemble_load_files) < 1:
            model.eval()
            with torch.no_grad():
                predictions = (model(Xinput)).cpu().numpy()
        else:
            predictions = np.empty((0, batch_size, config.n_output, Xinput.shape[-2], Xinput.shape[-1]))
            for model_save in model_ensemble_load_files:
                if os.path.exists(model_save):
                    print('in train: evaluation of model: ', model_save)
                    model_params_load(model_save, model, optimizer, config.DEVICE)
                    model.eval()

                    if config.MCdrop:
                        model.enable_dropout()
                        print(' running mcdrop iterations: ', config.MCdrop_it)
                        start_mcdropout_time = time.time()
                        for it in range(config.MCdrop_it):

                            with torch.no_grad():
                                out = to_np(model(Xinput))
                            
                            predictions = np.vstack((predictions, out[np.newaxis,...]))
                    else:
                        with torch.no_grad():
                            out = to_np(model(Xinput))
                        predictions = np.vstack((predictions, out[np.newaxis, ...]))

        output = get_impartial_outputs(predictions, config)  # output has keys: class_segmentation, factors

        if saveout:
            save_output_dic = config.basedir + config.model_name + '/output_images/' + str(epoch) + '/'
            file_output_save = 'eval_' + str(idx) + '.pickle'
            mkdir(save_output_dic)
            print('Saving output : ', save_output_dic + file_output_save)
            with open(save_output_dic + file_output_save, 'wb') as handle:
                pickle.dump(output, handle)

        idx += 1
        # if idx == 5:
        #     break
        output_list.append(output)

    return output_list, gt_list
