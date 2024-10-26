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
                mlflow.log_metric("train_loss_b", f"{loss_batch.item()}")
                losses_all.append(loss_batch.item())
                self.optimizer.step()


            logger.info("Train :::: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
            mlflow.log_metric("train_loss", f"{np.mean(losses_all):6f}")

            self.validate(dataloader_val, epoch=epoch)
            if epoch % 20 == 0:
                self.evaluate(dataloader_eval, epoch=epoch)
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

        logger.info("Val :::: Epoch: {} Loss: {}".format(epoch, np.mean(losses_all)))
        mlflow.log_metric("val_loss", f"{np.mean(losses_all):6f}")


    # infer == when there in no gt available
    def infer(self, dataloader_infer, epoch=0):
        
        logger.info("Start infer :::: ")
        print('Start inference in training ...')
        for batch, data in enumerate(dataloader_infer):

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
    def evaluate(self, dataloader_eval, epoch=0):
        
        print('Start evaluation in training ...')
        metrics_rows_pd = []
        for batch, data in enumerate(dataloader_eval):

            Xinput = data['input'].to(self.device)
            Ylabel = data['label'].numpy()[0,0,:,:].astype('int')
            image_name = data['image_name'][0]
            image_name = image_name.split('/')[-1]

            predictions = self.inference(Xinput)
            
            mean = True
            std = False
            out = get_impartial_outputs(predictions, self.classification_tasks, mean, std)  # output has keys: class_segmentation, factors
            out_seg = out['0']['class_segmentation'][0,0,:,:] 


            output_dir = os.path.join(self.output_dir, 'pred_w_gt')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.save_log_outputs(output_dir, image_name, epoch, predictions, out, out_seg, mlflow_tag='pred_w_gt')
            
            row = [image_name]
            metrics = get_performance(Ylabel, out_seg, threshold=0.5)
            for key in metrics.keys():
                row.append(metrics[key])
            metrics_rows_pd.append(row)

        columns = ['image_name']
        for key in metrics.keys():
            columns.append(key)

        model_output_pd_summary_path = os.path.join(self.output_dir, 'eval_{}.csv'.format(epoch))
        pd_summary = pd.DataFrame(data=metrics_rows_pd, columns=columns)
        pd_summary.to_csv(model_output_pd_summary_path, index=0) 

        mlflow.log_artifact(model_output_pd_summary_path, "evaluation")


    def save_log_outputs(self, output_dir, image_name, epoch, predictions, out, out_seg, mlflow_tag):
        
        png_components_path = os.path.join(output_dir, 'components_{}_{}.png'.format(image_name, epoch))
        # npz_prediction_path = os.path.join(output_dir, 'pred_{}_{}.npz'.format(image_name, epoch))
        # npz_impartial_outs_path = os.path.join(output_dir, 'out_{}_{}.npz'.format(image_name, epoch))
        png_prediction_path = os.path.join(output_dir, '{}_{}.png'.format(image_name, epoch))

        png_mask_path = os.path.join(output_dir, '{}_{}_mask.png'.format(image_name, epoch))
        roi_zip_path = os.path.join(output_dir, '{}_{}_roi.zip'.format(image_name, epoch))

        plot_impartial_outputs(out, png_components_path)
        # np.savez(npz_prediction_path, prediction=predictions)
        # np.savez(npz_impartial_outs_path, out=out)            
        plot_segmentation(out_seg, png_prediction_path)

        mlflow.log_artifact(png_prediction_path, mlflow_tag)
        mlflow.log_artifact(png_components_path, mlflow_tag)


        # threshold & save 
        threshold = 0.98
        out_mask = (out_seg > threshold).astype(np.uint8) * 255
        out_mask = Image.fromarray(out_mask)
        out_mask.save(png_mask_path)

        for contour in measure.find_contours((out_seg > threshold).astype(np.uint8), level=0.9999):
            roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
            roi.tofile(roi_zip_path)

        # mlflow.log_artifact(png_mask_path, mlflow_tag)
        # mlflow.log_artifact(roi_zip_path, mlflow_tag)
        

    # just the inference call w/ and wo/ mcdropout
    def inference(self, Xinput):
        self.model.eval()
        if self.mcdrop_it == 0:
            with torch.no_grad():
                predictions = to_np(self.model(Xinput))

        if self.mcdrop_it > 0:
            predictions = np.empty((0, Xinput.shape[0], self.n_output, Xinput.shape[-2], Xinput.shape[-1]))
            self.model.enable_dropout()
            print('Running MCDropout iterations: ', self.mcdrop_it)
            for it in range(self.mcdrop_it):
                with torch.no_grad():
                    out = to_np(self.model(Xinput))

                predictions = np.vstack((predictions, out[np.newaxis,...]))
            
        # print("mcdropout test: ", predictions.shape)
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

    # plt.save(output_file)

def plot_segmentation(prediction, output_file):
    plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.imshow(prediction)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()



def plot_impartial_outputs(out, output_file):
    plt.figure(figsize=(15,15))

    out = out['0']
    print(out.keys())
    print(out['factors'].keys())

    plt.subplot(4, 2, 1)
    plt.imshow(out['class_segmentation'][0,0,:,:])
    plt.subplot(4, 2, 2)
    plt.imshow(out['class_segmentation_variance'][0,0,:,:])
    
    output_factors = out['factors']
    plt.subplot(4, 2, 3)
    print("output_factors['components']: ", output_factors['components'].shape)
    plt.imshow(output_factors['components'][0,0,:,:])
    plt.subplot(4, 2, 4)
    print("output_factors['components_variance']: ", output_factors['components_variance'].shape)
    plt.imshow(output_factors['components_variance'][0,0,:,:])


    plt.subplot(4, 2, 5)
    plt.imshow(output_factors['mean_ch0'][0,0,:,:])
    plt.subplot(4, 2, 6)
    plt.imshow(output_factors['mean_variance_ch0'][0,0,:,:])

    plt.subplot(4, 2, 7)
    plt.imshow(output_factors['mean_ch1'][0,0,:,:])
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



# --output_dir experiments/impartial_vectra/ensemble_budget_0.4_scribble_test/