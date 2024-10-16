import sys
import os
import numpy as np
import logging
import mlflow 
import pandas as pd
from PIL import Image 

import torch
import matplotlib.pyplot as plt

from roifile import ImagejRoi
from skimage import measure

sys.path.append("../")
from general.utils import model_params_load, mkdir, to_np
from general.inference import get_impartial_outputs
from general.evaluation import get_performance

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
                losses_all.append(loss_batch.item())
                self.optimizer.step()


            logger.info("Train :::: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
            mlflow.log_metric("train_loss", f"{np.mean(losses_all):6f}")

            self.validate(dataloader_val, epoch=epoch)
            if epoch % 5 == 0:
                self.evaluate(dataloader_eval, epoch=epoch)
                self.infer(dataloader_infer, epoch=epoch)


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

        logger.info("Val :::: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
        mlflow.log_metric("val_loss", f"{np.mean(losses_all):6f}")


    def infer(self, dataloader_infer, epoch=0):
        
        logger.info("Start infer :::: ")
        output_list = []
        # gt_list = []
        print('Start inference in training ...')
        for batch, data in enumerate(dataloader_infer):

            Xinput = data['input'].to(self.device)
            image_name = data['image_name'][0]
            image_name = image_name.split('/')[-1]

            self.model.eval()
            if self.mcdrop_it == 0:
                with torch.no_grad():
                    predictions = (self.model(Xinput)).cpu().numpy()
                    # print("Eval: batch: predictions size: ", batch, predictions.shape)

            ### TODO: Add MCDropout in Infer & Validate
            if self.mcdrop_it > 0:
                predictions = np.empty((0, Xinput.shape[0], self.n_output, Xinput.shape[-2], Xinput.shape[-1]))
                self.model.enable_dropout()
                print('Running MCDropout iterations: ', self.mcdrop_it)
                for it in range(self.mcdrop_it):
                    with torch.no_grad():
                        out = to_np(self.model(Xinput))
                        # print("MCDrop out test: ", it, out.shape)

                    predictions = np.vstack((predictions, out[np.newaxis,...]))
            
            print("mcdropout test: ", predictions.shape)

            output_file = os.path.join(self.output_dir, '{}_{}.png'.format(epoch, batch))
            # plot_save(predictions, output_file)

            npz_prediction_path_dir = os.path.join(self.output_dir, 'inference')
            if not os.path.exists(npz_prediction_path_dir):
                os.makedirs(npz_prediction_path_dir)
            
            npz_prediction_path = os.path.join(npz_prediction_path_dir, '{}_{}_{}.npz'.format(image_name, epoch, batch))
            np.savez(npz_prediction_path, prediction=predictions)
            
            mean = True
            std = False
            out = get_impartial_outputs(predictions, self.classification_tasks, mean, std)  # output has keys: class_segmentation, factors
            out_seg = out['0']['class_segmentation'][0,0,:,:]
            
            png_prediction_path = os.path.join(npz_prediction_path_dir, '{}_{}_{}.png'.format(image_name, epoch, batch))

            # plot_predictions(data, out_seg, png_prediction_path)
            plot_segmentation(out_seg, png_prediction_path)


            # threshold & save 
            threshold = 0.98
            out_mask = (out_seg > threshold).astype(np.uint8) * 255
            png_mask_path = os.path.join(npz_prediction_path_dir, '{}_{}_{}_mask.png'.format(image_name, epoch, batch))
            out_mask = Image.fromarray(out_mask)
            out_mask.save(png_mask_path)


            roi_zip_path = os.path.join(npz_prediction_path_dir, '{}_{}_{}_roi.zip'.format(image_name, epoch, batch))
            for contour in measure.find_contours((out_seg > threshold).astype(np.uint8), level=0.9999):
                roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
                roi.tofile(roi_zip_path)

            mlflow.log_artifact(png_prediction_path, "inference")
            mlflow.log_artifact(png_mask_path, "inference")
            mlflow.log_artifact(roi_zip_path, "inference")

        return output_list
    

    def evaluate(self, dataloader_eval, epoch=0):
        
        logger.info("Start eval :::: ")
        pd_rows = []
 
        print('Start evaluation in training ...')
        for batch, data in enumerate(dataloader_eval):

            Xinput = data['input'].to(self.device)
            Ylabel = data['label'].numpy()[0,0,:,:].astype('int')
            image_name = data['image_name'][0]
            image_name = image_name.split('/')[-1]

            self.model.eval()
            if self.mcdrop_it == 0:
                with torch.no_grad():
                    predictions = (self.model(Xinput)).cpu().numpy()
                    # print("Eval: batch: predictions size: ", batch, predictions.shape)

            ### TODO: Add MCDropout in Infer & Validate
            if self.mcdrop_it > 0:
                predictions = np.empty((0, Xinput.shape[0], self.n_output, Xinput.shape[-2], Xinput.shape[-1]))
                self.model.enable_dropout()
                print('Running MCDropout iterations: ', self.mcdrop_it)
                for it in range(self.mcdrop_it):
                    with torch.no_grad():
                        out = to_np(self.model(Xinput))
                        # print("MCDrop out test: ", it, out.shape)

                    predictions = np.vstack((predictions, out[np.newaxis,...]))
            
            print("mcdropout test: ", predictions.shape)


            output_file = os.path.join(self.output_dir, '{}_{}.png'.format(epoch, batch))
            # plot_save(predictions, output_file)

            npz_prediction_path_dir = os.path.join(self.output_dir, 'predictions')
            if not os.path.exists(npz_prediction_path_dir):
                os.makedirs(npz_prediction_path_dir)
            
            npz_prediction_path = os.path.join(npz_prediction_path_dir, '{}_{}.npz'.format(epoch, batch))
            np.savez(npz_prediction_path, prediction=predictions)

            mean = True
            std = False
            out = get_impartial_outputs(predictions, self.classification_tasks, mean, std)  # output has keys: class_segmentation, factors
            out_seg = out['0']['class_segmentation'][0,0,:,:]
            # print("GS:::eval debug out_seg shape", out_seg.shape)
            png_prediction_path = os.path.join(npz_prediction_path_dir, '{}_{}_{}.png'.format(image_name, epoch, batch))

            # plot_predictions(data, out_seg, png_prediction_path)
            plot_segmentation(out_seg, png_prediction_path)
            mlflow.log_artifact(png_prediction_path, "predictions")

            
            row = [image_name]
            print("GS:::eval debug Ylabel shape", Ylabel.shape)

            metrics = get_performance(Ylabel, out_seg, threshold=0.5)
            for key in metrics.keys():
                row.append(metrics[key])
            pd_rows.append(row)

        columns = ['image_name']
        for key in metrics.keys():
            columns.append(key)

        model_output_pd_summary_path = os.path.join(self.output_dir, 'eval_{}.csv'.format(epoch))
        pd_summary = pd.DataFrame(data=pd_rows, columns=columns)
        pd_summary.to_csv(model_output_pd_summary_path, index=0) 

        mlflow.log_artifact(model_output_pd_summary_path, "evaluation")

    

def plot_save(predictions, output_file):

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

    plt.savefig(output_file)
    plt.close()

    # plt.save(output_file)

def plot_segmentation(prediction, output_file):
    plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.imshow(prediction)
    plt.savefig(output_file)
    plt.close()

