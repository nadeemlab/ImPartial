
import sys
sys.path.append("../")
from general.utils import TravellingMean,to_np
import numpy as np
import torch.nn as nn
import time

def epoch_recseg_loss(dataloader, model, optimizer, config, criterio,train_type=True):
    #criterio has to output loss, seg_fore loss, seg_back loss and rec loss

    if train_type:
        model.train()
    else:
        model.eval()

    output = {}
    output['loss'] = TravellingMean()
    for key in config.weight_objectives.keys():
        output[key] = TravellingMean()

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
            optimizer.step()

        # tic_list.append(time.perf_counter())
        # tic_final_list.append(tic_list[-1] - tic_list[-2])
        # print(' losses etc : ', tic_list[-1] - tic_list[-2])

    # print('Total time (batch, criterio, backward, total) : ')
    # print(np.round(np.mean(np.array(tic_bs_list)),3),
    #       np.round(np.mean(np.array(tic_criterio_list)),3),
    #       np.round(np.mean(np.array(tic_final_list)), 3),np.round(tic_list[-1]-tic_list[0],1))
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


# def epoch_recseg_loss(dataloader, model, optimizer, config, criterio,train_type=True):
#     #criterio has to output loss, seg_fore loss, seg_back loss and rec loss
#
#     if train_type:
#         model.train()
#     else:
#         model.eval()
#
#     output = {}
#     output['loss'] = TravellingMean()
#     output['rec'] = TravellingMean()
#     output['seg_fore'] = TravellingMean()
#     output['seg_back'] = TravellingMean()
#
#     npatches_total = 0 #counter to consider epoch done
#     npatches_step = 0 #counter to do sgd update
#
#     tic_list = []
#     while npatches_total < config.npatches_epoch: #epoch is determined by the number of patches that we sampled
#
#         tic_list.append(time.perf_counter())
#         for batch, data in enumerate(dataloader):
#             x_b = data['input'].flatten(start_dim = 0,end_dim = 1)#.to(config.DEVICE)
#             mask_b = data['mask'].flatten(start_dim = 0,end_dim = 1)#.to(config.DEVICE)
#             scribble_b = data['scribble'].flatten(start_dim = 0,end_dim = 1)#.to(config.DEVICE)
#
#             if npatches_step == 0:
#                 x = x_b
#                 mask = mask_b
#                 scribble = scribble_b
#             else:
#                 x = torch.cat((x,x_b),0)
#                 mask = torch.cat((mask, mask_b), 0)
#                 scribble = torch.cat((scribble, scribble_b), 0)
#
#             npatches_step += x_b.shape[0]
#
#             # evaluate when the number of patches per batch was reached
#             if (npatches_step >= config.npatches_sgd_step):
#                 tic_list.append(time.perf_counter())
#                 print(' batch sampling : ', tic_list[-1] - tic_list[-2])
#
#                 npatches_total += npatches_step
#                 npatches_step = 0
#
#                 x = x.to(config.DEVICE)
#                 mask = mask.to(config.DEVICE)
#                 scribble = scribble.to(config.DEVICE)
#
#                 out = model(x)
#                 losses = criterio(out, x, scribble, mask)
#
#                 tic_list.append(time.perf_counter())
#                 print(' criterio : ', tic_list[-1] - tic_list[-2])
#
#                 loss_batch = 0
#                 for key in ['rec','seg_fore','seg_back']:
#                     loss_batch += losses[key] * config.weight_objectives[key]
#
#                 # total_loss = (total_loss*npatches + loss_batch*x.shape[0])/(npatches + x.shape[0])
#                 # if train_type:
#                 #     optimizer.virtual_step()
#
#                 output['loss'].update(to_np(loss_batch), mass=x.shape[0])
#                 output['rec'].update(to_np(losses['rec']), mass=x.shape[0])
#                 output['seg_fore'].update(to_np(losses['seg_fore']), mass=x.shape[0])
#                 output['seg_back'].update(to_np(losses['seg_back']), mass=x.shape[0])
#
#                 npatches_total += x.shape[0]
#
#                 if train_type:
#                     optimizer.zero_grad()
#                     loss_batch.backward()
#                     optimizer.step()
#
#                 tic_list.append(time.perf_counter())
#                 print(' end : ', tic_list[-1] - tic_list[-2])
#
#                     # break
#
#     # print(total_loss)
#     for key in output.keys():
#         output[key] = output[key].mean
#     return output
