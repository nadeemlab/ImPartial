
import sys
sys.path.append("../")
from general.utils import TravellingMean,to_np
import numpy as np
import torch.nn as nn
import time
from torch import optim

def epoch_recseg_loss(dataloader, model, optimizer, config, criterio,train_type=True):
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

        out = model(x)
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
            if config.max_grad_clip>0:
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


def recseg_trainer(dataloader_train,dataloader_val,model,optimizer,criterio,config):
    from general.training import epoch_recseg_loss
    from general.utils import early_stopping, model_params_save, model_params_load

    history = {}

    history['loss_mbatch_train'] = []

    for key in config.weight_objectives.keys():
        history[key+'_mbatch_train'] = []

    tags_plot = ['loss']
    for key in config.weight_objectives.keys():
        tags_plot.append(key)
    for _ in ['train', 'val']:
        history['loss_' + _] = []
        for key in config.weight_objectives.keys():
            history[key + '_' + _] = []


    stopper = early_stopping(config.patience, 0, np.infty)

    patch_counter = 0
    for epoch in range(config.EPOCHS):

        # Training:
        output = epoch_recseg_loss(dataloader_train, model, optimizer, config,
                                   criterio, train_type=True)
        for key in output.keys():
            history[key + '_mbatch_train'].append(output[key])

        # Evaluation!
        # output = epoch_recseg_loss(dataloader_train, model, optimizer, config,
                                   # criterio, train_type=False)
        for key in output.keys():
            history[key + '_train'].append(output[key])
        output = epoch_recseg_loss(dataloader_val, model, optimizer, config,
                                   criterio, train_type=False)
        for key in output.keys():
            history[key + '_val'].append(output[key])

        model_params_save(config.basedir + config.model_name + '/' + config.last_model,
                          model, optimizer)  # save last model

        if config.val_stopper:
            save, stop = stopper.evaluate(history['loss_val'][-1])
        else:
            save, stop = stopper.evaluate(history['loss_train'][-1])

        if save:
            history['epoch_best'] = epoch
            print('saving best model, epoch: ', epoch)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model,
                              model, optimizer)  # save best model

        string_print = 'epoch : ' + str(epoch)
        for key in tags_plot:
            string_print = string_print + ' | ' + key + ' mbtr, tr,val : ' + str(
                np.round(history[key + '_mbatch_train'][-1], 3)) + ','
            string_print = string_print + str(np.round(history[key + '_train'][-1], 3)) + ','
            string_print = string_print + str(np.round(history[key + '_val'][-1], 3))
        string_print = string_print + ' | stop_pat :' + str(stopper.counter)
        print(string_print)

        if epoch < config.warmup_epochs:
            stop = False  # stop = false if we are still in the warmup epochs

        if stop:
            break

        patch_counter += 1
        if patch_counter == config.nepochs_sample_patches:
            print('sampling new patches...')
            dataloader_train.dataset.sample_patches_data(npatches_total=config.npatches_epoch)
            dataloader_val.dataset.sample_patches_data(npatches_total=config.npatches_epoch)
            patch_counter = 0

    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + config.best_model)
    model_params_load(config.basedir + config.model_name + '/' + config.best_model, model, optimizer, config.DEVICE)
    return history



# def recseg_multimodelsave_trainer(dataloader_train,dataloader_val,model,optimizer,criterio,config,nsaves = 5):
#     from general.training import epoch_recseg_loss
#     from general.utils import early_stopping, model_params_save, model_params_load
#
#     val_model_saves_list = ['model_val_best_save'+str(i)+'.pth' for i in range(nsaves)]
#     val_epochs_saves_list = [0 for i in range(nsaves)]
#     train_model_saves_list = ['model_train_best_save' + str(i) + '.pth' for i in range(nsaves)]
#     train_epochs_saves_list = [0 for i in range(nsaves)]
#
#     history = {}
#
#     history['loss_mbatch_train'] = []
#     for key in config.weight_objectives.keys():
#         history[key+'_mbatch_train'] = []
#     history['grad_mbatch_train'] = []
#
#     tags_plot = ['loss']
#     for key in config.weight_objectives.keys():
#         tags_plot.append(key)
#     for _ in ['train', 'val']:
#         history['loss_' + _] = []
#         for key in config.weight_objectives.keys():
#             history[key + '_' + _] = []
#
#     stopper = early_stopping(config.patience, 0, np.infty)
#     stopper_train = early_stopping(config.patience, 0, np.infty)
#     stopper_val = early_stopping(config.patience, 0, np.infty)
#
#     patch_counter = 0
#
#     ix_val_save = 0
#     ix_train_save = 0
#
#     for epoch in range(config.EPOCHS):
#
#         # Training:
#         output = epoch_recseg_loss(dataloader_train, model, optimizer, config,
#                                    criterio, train_type=True)
#         for key in output.keys():
#             history[key + '_mbatch_train'].append(output[key])
#
#         # Evaluation!
#         # output = epoch_recseg_loss(dataloader_train, model, optimizer, config,
#                                    # criterio, train_type=False)
#         for key in output.keys():
#             if key != 'grad':
#                 history[key + '_train'].append(output[key])
#         output = epoch_recseg_loss(dataloader_val, model, optimizer, config,
#                                    criterio, train_type=False)
#         for key in output.keys():
#             if key != 'grad':
#                 history[key + '_val'].append(output[key])
#
#         # model_params_save(config.basedir + config.model_name + '/' + config.last_model,
#                           # model, optimizer)  # save last model
#
#
#
#
#         if config.val_stopper:
#             save, stop = stopper.evaluate(history['loss_val'][-1])
#         else:
#             save, stop = stopper.evaluate(history['loss_train'][-1])
#
#         save_train, _ = stopper_train.evaluate(history['loss_train'][-1])
#         save_val, _ = stopper_val.evaluate(history['loss_val'][-1])
#
#         if save:
#             history['epoch_best'] = epoch
#             # model_params_save(config.basedir + config.model_name + '/' + val_model_saves_list[ix_val_save],
#                               # model, optimizer)  # save best model
#             # val_epochs_saves_list[ix_val_save] = epoch
#             # ix_val_save += 1
#             # if ix_val_save >= nsaves:
#             #     ix_val_save = 0
#
#         if save_train:
#             model_params_save(config.basedir + config.model_name + '/' + train_model_saves_list[ix_train_save],
#                               model, optimizer)  # save best model
#             train_epochs_saves_list[ix_train_save] = epoch
#
#         if save_val:
#             print('saving best model, epoch: ', epoch)
#             model_params_save(config.basedir + config.model_name + '/' + val_model_saves_list[ix_val_save],
#                               model, optimizer)  # save best model
#             val_epochs_saves_list[ix_val_save] = epoch
#
#         string_print = 'epoch : ' + str(epoch)
#         for key in tags_plot:
#             string_print = string_print + ' | ' + key + ' mbtr, tr,val : ' + str(
#                 np.round(history[key + '_mbatch_train'][-1], 3)) + ','
#             string_print = string_print + str(np.round(history[key + '_train'][-1], 3)) + ','
#             string_print = string_print + str(np.round(history[key + '_val'][-1], 3))
#         string_print = string_print + ' | stop_pat :' + str(stopper.counter)
#         print(string_print)
#         print(val_epochs_saves_list)
#         print(train_epochs_saves_list)
#         print()
#
#         if epoch < config.warmup_epochs:
#             stop = False  # stop = false if we are still in the warmup epochs
#
#         if stop:
#             break
#
#         patch_counter += 1
#         if patch_counter == config.nepochs_sample_patches:
#             print('sampling new patches...')
#             dataloader_train.dataset.sample_patches_data(npatches_total=config.npatches_epoch)
#             dataloader_val.dataset.sample_patches_data(npatches_total=config.npatches_epoch)
#             patch_counter = 0
#
#             #reset train stopper and increase train save, note that we save once per sample patches iteration
#             stopper_train = early_stopping(config.patience, 0, np.infty)
#             ix_train_save += 1
#             if ix_train_save >= nsaves:
#                 ix_train_save = 0
#
#             stopper_val = early_stopping(config.patience, 0, np.infty)
#             ix_val_save += 1
#             if ix_val_save >= nsaves:
#                 ix_val_save = 0
#
#     history['val_epochs_saves_list'] = val_epochs_saves_list
#     history['train_epochs_saves_list'] = train_epochs_saves_list
#
#     history['val_model_saves_list'] = val_model_saves_list
#     history['train_model_saves_list'] = train_model_saves_list
#
#     print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + val_model_saves_list[ix_val_save])
#     model_params_load(config.basedir + config.model_name + '/' + val_model_saves_list[ix_val_save], model, optimizer, config.DEVICE)
#     return history
#


def recseg_checkpoint_ensemble_trainer(dataloader_train,dataloader_val,model,optimizer,criterio,config):

    from general.training import epoch_recseg_loss
    from general.utils import early_stopping, model_params_save, model_params_load

    ## Savings
    history = {}
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
    stopper = early_stopping(config.patience, 0, np.infty)

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

        #Training:
        output = epoch_recseg_loss(dataloader_train, model, optimizer, config,
                                   criterio, train_type=True)

        for key in output.keys():
            history[key + '_mbatch_train'].append(output[key])

        # tic_list.append(time.perf_counter())
        # print( 'epoch train : ',tic_list[-1] - tic_list[-2])

        output = epoch_recseg_loss(dataloader_val, model, optimizer, config,
                                   criterio, train_type=False)

        for key in output.keys():
            if key != 'grad':
                history[key + '_val'].append(output[key])

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
            model_params_save(config.basedir + config.model_name + '/' + model_saves_list[patch_sampler],model, optimizer)  # save best model
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