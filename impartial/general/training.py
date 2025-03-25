import sys
import os
import numpy as np
import logging
import pickle
import mlflow 
import pandas as pd
from PIL import Image 

import torch
import matplotlib.pyplot as plt

from roifile import ImagejRoi
from skimage import measure

import skimage
from scipy import ndimage
from impartial.general.outlines import dilate_masks, masks_to_outlines

from impartial.general.utils import model_params_save, to_np, early_stopping
from impartial.general.inference import get_impartial_outputs, get_entropy
from impartial.general.evaluation import get_performance

import logging
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, device, classification_tasks, model, criterion, optimizer, output_dir, n_output, epochs, mcdrop_it=0):
        self.epochs = epochs
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model 
        self.output_dir = output_dir 
        self.classification_tasks = classification_tasks

        self.mcdrop_it = mcdrop_it
        self.n_output = n_output 

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, dataloader_train, dataloader_val, dataloader_eval, dataloader_infer):
        print("Start training ... ")
        
        patience = 10
        stopper = early_stopping(patience, 0, np.inf) #only to save global best model
        best_model_path = os.path.join(self.output_dir, "model_best.pth")
        
        losses_all = []
        for epoch in range(1, self.epochs):
            
            for batch, data in enumerate(dataloader_train):

                x = data['input'].to(self.device) #input image with blind spots replaced randomly
                mask = data['mask'].to(self.device)
                scribble = data['scribble'].to(self.device)
                target = data['target'].to(self.device) #input image with non blind spots

                # print(x.size(), mask.size(), scribble.size(), target.size())

                out = self.model(x)
                losses = self.criterion.compute_loss(out, target, scribble, mask)
                
                loss_batch = 0
                
                # TODO: check weighted loss 
                for key in self.criterion.config.weight_objectives.keys():
                    loss_batch += losses[key] * self.criterion.config.weight_objectives[key]
                    if torch.is_tensor(losses[key]):
                        mlflow.log_metric(key, losses[key].item())
                    else:
                        mlflow.log_metric(key, losses[key])

                self.optimizer.zero_grad()

                loss_batch.backward()

                logger.debug("Epoch: {} Batch: {} Loss: {}".format(epoch, batch, loss_batch.item()))
                mlflow.log_metric("train_loss_b", f"{loss_batch.item()}")
                losses_all.append(loss_batch.item())
                self.optimizer.step()


            logger.info("Train :::: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
            mlflow.log_metric("train_loss", f"{np.mean(losses_all):6f}")

            loss_mean = self.validate(dataloader_val, epoch=epoch)
            
            is_save, _ = stopper.evaluate(loss_mean)
            if is_save:
                checkpoint_model_path = os.path.join(self.output_dir, 'checkpoint_{}'.format(epoch))
                logger.info("Saving the current best model: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
                model_params_save(best_model_path, self.model, self.optimizer)  # save best model
                model_params_save(checkpoint_model_path, self.model, self.optimizer)  # save best model
                mlflow.log_artifact(best_model_path, "model")


            if epoch % 20 == 0:
                # self.evaluate(dataloader_eval, epoch=epoch, eval_freq=50, is_save=True, dilate=True)
                # self.evaluate(dataloader_eval, epoch=epoch, eval_freq=1, is_save=True, dilate=True)
                # self.evaluate(dataloader_eval, epoch=epoch, eval_freq=50, is_save=True, dilate=True)
                self.evaluate(dataloader_eval, epoch=epoch, eval_freq=50, is_save=False, dilate=True)
            # if epoch % 50 == 0:
            #     self.infer(dataloader_infer, epoch=epoch)


    def validate(self, dataloader_val, epoch=0):
        losses_all = []
        for batch, data in enumerate(dataloader_val):

            x = data['input'].to(self.device) #input image with blind spots replaced randomly
            mask = data['mask'].to(self.device)
            scribble = data['scribble'].to(self.device)
            target = data['target'].to(self.device) #input image with non blind spots

            out = self.model(x)
            losses = self.criterion.compute_loss(out, target, scribble, mask)

            loss_batch = 0
            # TODO: check weighted loss 
            for key in self.criterion.config.weight_objectives.keys():
                loss_batch += losses[key] * self.criterion.config.weight_objectives[key]

            losses_all.append(loss_batch.item())

        loss_mean = np.mean(losses_all)
        logger.info("Val :::: Epoch: {} Loss: {}".format(epoch, loss_mean))
        mlflow.log_metric("val_loss", f"{loss_mean:6f}")
        
        return loss_mean 
    

    # infer == when there in no gt available
    def infer(self, dataloader_infer, epoch=0):
        
        logger.info("Start infer :::: ")
        print('Start inference in training ...')
        for batch, data in enumerate(dataloader_infer):
            logger.info("Infer: batch: {} / {}  mcdropout: {}".format(batch, len(dataloader_infer)), self.mcdrop_it)
            Xinput = data['input'].to(self.device)
            image_name = data['image_name'][0]
            image_name = image_name.split('/')[-1]

            predictions = self.inference(Xinput)

            mean = True
            std = False
            out = get_impartial_outputs(predictions, self.classification_tasks, mean, std)  # output has keys: class_segmentation, factors
            out_seg = out['0']['class_segmentation'][0,0,:,:]

            output_dir = os.path.join(self.output_dir, 'pred_no_gt')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.save_log_outputs(output_dir, image_name, epoch, predictions, out, out_seg, mlflow_tag='pred_no_gt')
            

    # evaluate == when there in gt available
    def evaluate(self, dataloader_eval, epoch=0, eval_freq=1, is_save=False, dilate=False):
        
        print('Start evaluation in training ...')
        logger.info('Start evaluation in training ...')
        metrics_rows_pd = []
        for batch, data in enumerate(dataloader_eval):
            logger.info("Eval: batch: {} / {}  mcdropout: {}".format(batch, len(dataloader_eval), self.mcdrop_it))
            Xinput = data['input'].to(self.device)
            Ylabel = data['label'].numpy()[0,0,:,:].astype('int')
            image_name = data['image_name'][0]
            image_name = image_name.split('/')[-1]

            predictions = self.inference(Xinput) # TODO: Change
            # predictions = self.inference_tiled(Xinput)
            
            mean = True
            std = False
            out = get_impartial_outputs(predictions, self.classification_tasks, mean, std)  # output has keys: class_segmentation, factors
            out_seg = out['0']['class_segmentation'][0,0,:,:] 


            output_dir = os.path.join(self.output_dir, 'pred_w_gt')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # TODO: Only save a few samples
            if is_save and batch % eval_freq == 0:
                self.save_log_outputs(output_dir, image_name, epoch, predictions, out, out_seg, mlflow_tag='pred_w_gt')
                self.save_out_seg_npz(output_dir, image_name, epoch, out_seg, mlflow_tag='npz_pred')
                self.save_label_gt(output_dir, image_name, epoch, Ylabel, mlflow_tag='label_gt')
                
            row = [image_name]
            metrics = get_performance(Ylabel, out_seg, threshold=0.5, dilate=dilate)
            for key in metrics.keys():
                row.append(metrics[key])
            metrics_rows_pd.append(row)

        columns = ['image_name']
        for key in metrics.keys():
            columns.append(key)

        model_output_pd_summary_path = os.path.join(self.output_dir, 'eval_{}.csv'.format(epoch))
        model_mean_pd_summary_path = os.path.join(self.output_dir, 'eval_mean_{}.csv'.format(epoch))
        pd_summary = pd.DataFrame(data=metrics_rows_pd, columns=columns)
        pd_summary.to_csv(model_output_pd_summary_path, index=0) 

        pd_summary_mean = pd_summary.mean()
        pd_summary_mean.to_csv(model_mean_pd_summary_path) 
        
        mlflow.log_artifact(model_output_pd_summary_path, "evaluation")
        mlflow.log_artifact(model_mean_pd_summary_path, "evaluation")


    def save_out_seg_npz(self, output_dir, image_name, epoch, out_seg, mlflow_tag):
        
        npz_out_seg_path = os.path.join(output_dir, 'out_seg_{}_{}.npz'.format(image_name, epoch))
        np.savez(npz_out_seg_path, out=out_seg)            

        mlflow.log_artifact(npz_out_seg_path, mlflow_tag)
        
    def save_label_gt(self, output_dir, image_name, epoch, label_gt, mlflow_tag):
        
        png_lablel_gt_path = os.path.join(output_dir, 'label_{}_{}.png'.format(image_name, epoch))
        out_mask = (label_gt > 0.5).astype(np.uint8) * 255
        out_mask = Image.fromarray(out_mask)
        out_mask.save(png_lablel_gt_path)

        mlflow.log_artifact(png_lablel_gt_path, mlflow_tag)
        

    def save_log_outputs(self, output_dir, image_name, epoch, predictions, out, out_seg, mlflow_tag):
        
        png_components_path = os.path.join(output_dir, 'components_{}_{}.png'.format(image_name, epoch))
        # npz_prediction_path = os.path.join(output_dir, 'pred_{}_{}.npz'.format(image_name, epoch))
        npz_impartial_outs_path = os.path.join(output_dir, 'out_{}_{}.pickle'.format(image_name, epoch))
        png_prediction_path = os.path.join(output_dir, '{}_{}.png'.format(image_name, epoch))

        png_mask_path = os.path.join(output_dir, '{}_{}_mask.png'.format(image_name, epoch))
        png_outlines_path = os.path.join(output_dir, '{}_{}_outlines.png'.format(image_name, epoch))
        roi_zip_path = os.path.join(output_dir, '{}_{}_roi.zip'.format(image_name, epoch))

        plot_impartial_outputs(out, png_components_path) # uncomment
        # np.savez(npz_prediction_path, prediction=predictions)
        
        with open(npz_impartial_outs_path, 'wb') as handle:
            pickle.dump(out, handle)

        plot_segmentation(out_seg, png_prediction_path)

        mlflow.log_artifact(png_prediction_path, mlflow_tag)
        mlflow.log_artifact(png_components_path, mlflow_tag) # uncomment


        # threshold & save 
        threshold = 0.95
        # threshold = 0.70
        out_mask = (out_seg > threshold).astype(np.uint8) * 255
        out_mask = Image.fromarray(out_mask)
        out_mask.save(png_mask_path)
        
        labels_pred, _ = ndimage.label(out_mask)
        labels_pred = skimage.morphology.remove_small_objects(labels_pred, min_size=5)
        labels_pred = dilate_masks(labels_pred, n_iter=1)
        mask_outlines = masks_to_outlines(labels_pred)

        noutX, noutY = np.nonzero(mask_outlines)
        mask_outlines_img = np.zeros((out_mask.height, out_mask.width, 3), dtype=np.uint8)
        mask_outlines_img[noutX, noutY] = np.array([255, 0, 0])
        mask_outlines_img = Image.fromarray(mask_outlines_img)
        mask_outlines_img.save(png_outlines_path)

        for contour in measure.find_contours((out_seg > threshold).astype(np.uint8), level=0.9999):
            roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
            roi.tofile(roi_zip_path)

        mlflow.log_artifact(png_mask_path, mlflow_tag)
        mlflow.log_artifact(png_outlines_path, mlflow_tag)
        mlflow.log_artifact(roi_zip_path, mlflow_tag)
        

    # just the inference call w/ and wo/ mcdropout
    def inference(self, Xinput):
        self.model.eval()
        if self.mcdrop_it == 0:
            with torch.no_grad():
                predictions = to_np(self.model(Xinput))

        if self.mcdrop_it > 0:
            predictions = np.empty((0, Xinput.shape[0], self.n_output, Xinput.shape[-2], Xinput.shape[-1]))
            self.model.enable_dropout()
            # print('Running MCDropout iterations: ', self.mcdrop_it)
            for it in range(self.mcdrop_it):
                with torch.no_grad():
                    out = to_np(self.model(Xinput))

                predictions = np.vstack((predictions, out[np.newaxis,...]))
            
        # print("mcdropout test: ", predictions.shape)
        return predictions


    def inference_tiled(self, Xinput, tile_size=256, stride=144):
        self.model.eval()

        height, width = 400, 400

        pred_map = torch.zeros(size=(1, 12, 400, 400), dtype=torch.float32)
        count_map = torch.zeros(size=(1, 12, 400, 400), dtype=torch.float32)

        # print("pred_map.shape: ", pred_map.shape)
        count = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # print("x, y :", x, y, count)
                count += 1
                tile = Xinput[:, :, y:y + tile_size, x:x + tile_size]
                # print("tile.shape: ", tile.shape)
                # print("pred_map[:, :, y:y + tile_size, x:x + tile_size]: ", pred_map[:, :, y:y + tile_size, x:x + tile_size].shape)

                with torch.no_grad():
                    pred_tile = to_np(self.model(tile))  # Add batch dimension
                    # print("pred_tile.shape: ", pred_tile.shape)

                pred_map[:, :, y:y + tile_size, x:x + tile_size] += pred_tile
                count_map[:, :, y:y + tile_size, x:x + tile_size] += 1

        # print("tile prediction: count", count)
        # Avoid division by zero and normalize by count map
        count_map[count_map == 0] = 1  # to avoid division by zero
        predictions = pred_map / count_map

        return predictions



def plot_save_predictions(predictions, output_file):

    plt.figure(figsize=(10,10))
    n = predictions.shape[1]
    
    # for i in range(n):
    #     plt.subplot(1,12,i+1)
    #     plt.imshow(predictions[0,i,:,:])

    # plt.savefig(output_file)
    # plt.save(output_file)
    for i  in range(0,3) :

        plt.subplot(3,4,1+(i*4))
        plt.imshow(predictions[0,0+(i*4),:,:])

        plt.subplot(3,4,2+(i*4))
        plt.imshow(predictions[0,1+(i*4),:,:])

        plt.subplot(3,4,3+(i*4))
        plt.imshow(predictions[0,2+(i*4),:,:])

        plt.subplot(3,4,4+(i*4))
        plt.imshow(predictions[0,3+(i*4),:,:])

        plt.show()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_segmentation(prediction, output_file):
    plt.figure(figsize=(10, 10))
    plt.subplot(1,2,1)
    plt.imshow(prediction)
    plt.subplot(1,2,2)
    plt.imshow(get_entropy(prediction))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_impartial_outputs(out, output_file):
    plt.figure(figsize=(15,15))

    out = out['0']
    # print(out.keys())
    # print(out['factors'].keys())

    plt.subplot(4, 2, 1)
    plt.imshow(out['class_segmentation'][0,0,:,:])
    if 'class_segmentation_variance' in out:
        plt.subplot(4, 2, 2)
        plt.imshow(out['class_segmentation_variance'][0,0,:,:])
    
    output_factors = out['factors']
    plt.subplot(4, 2, 3)
    # print("output_factors['components']: ", output_factors['components'].shape)
    plt.imshow(output_factors['components'][0,0,:,:])
    
    if 'components_variance' in output_factors:
        plt.subplot(4, 2, 4)
        # print("output_factors['components_variance']: ", output_factors['components_variance'].shape)
        plt.imshow(output_factors['components_variance'][0,0,:,:])

    if 'mean_ch0' in output_factors:
        plt.subplot(4, 2, 5)
        plt.imshow(output_factors['mean_ch0'][0,0,:,:])
    if 'mean_variance_ch0' in output_factors:
        plt.subplot(4, 2, 6)
        plt.imshow(output_factors['mean_variance_ch0'][0,0,:,:])

    if 'mean_ch1' in output_factors:
        plt.subplot(4, 2, 7)
        plt.imshow(output_factors['mean_ch1'][0,0,:,:])
        
    if 'mean_variance_ch1' in output_factors:
        plt.subplot(4, 2, 8)
        plt.imshow(output_factors['mean_variance_ch1'][0,0,:,:])


    # keys = ['class_segmentation', 'class_segmentation_variance']
    # keys += ['components', 'components_variance']
    # keys += ['mean_ch_0', 'mean_variance_ch_0']
    # keys += ['mean_ch_1', 'mean_variance_ch_1']
    # # keys += ['logstd_ch_0', 'logstd_variance_ch_0']
    # # keys += ['logstd_ch_1', 'logstd_variance_ch_1']
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
