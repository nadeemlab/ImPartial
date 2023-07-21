import torch
from scipy import ndimage
import sys
import os
import numpy as np
from torchvision import transforms

sys.path.append('../')

from general.networks import UNet
from general.evaluation import get_performance
from segmentation.dataset import ImageSegDatasetTiff
from segmentation.datalist import train_data, test_data
from segmentation.losses import FocalLossBinary 
from segmentation.transforms import Compose, Resize, RandomResize, RandomCrop, RandomHorizontalFlip, ToTensor


# 1. Model
model = UNet(
    n_channels=2,
    n_classes=1,
    depth=4,
    base=32,
    activation='relu',
    batchnorm=False,
    dropout=False,
    dropout_lastconv=False,
    p_drop=0.5
)
model = model.to(device='cuda')

# 2. Optimizer
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# 3. Dataset
data_dir = "/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/"
# data_dir = "/nadeem_lab/Gunjan/downloads/Vectra_WC_2CH_tiff_full_labels/"
# data_dir = "/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/"
# data_dir = "/Users/gshrivastava/Mskcc/Vectra_WC_2CH_tiff_full_labels/"

transform_train = [
                    transforms.ToTensor(),
                  ]
transform_train = transforms.Compose(transform_train)

transform_test = [
                    transforms.ToTensor(),
                  ]
transform_test = transforms.Compose(transform_test)

# # 3.1 dataset and dataloaders
# dataset_files = {} 
# dataset_files['images'] = image_list
# dataset_files['labels'] = label_list

image_list, label_list = train_data(data_dir)

image_train_list = image_list[:300]
label_train_list = label_list[:300]
dataset_train = ImageSegDatasetTiff(data_dir, image_train_list, label_train_list, transform_train)

image_val_list = image_list[300:]
label_val_list = label_list[300:]
dataset_val = ImageSegDatasetTiff(data_dir, image_val_list, label_val_list, transform_test)


# (2) Make dataset test 
# dataset_test = CPTissuenetDatasetTiff(data_dir, transform_test)
data_test_dir = "/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/"
image_test_list, label_test_list = test_data(data_test_dir)
dataset_test = ImageSegDatasetTiff(data_dir, image_test_list, label_test_list, transform_test)

print("Size dataset_train: ", len(dataset_train))
print("Size dataset_val: ", len(dataset_val))
print("Size dataset_test: ", len(dataset_test))

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=32)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)


# 4. Loss Function
# criterio = torch.nn.BCEWithLogitsLoss()
criterio = FocalLossBinary()

# 5. Train function 
def train(model, criterio, optimizer, dataloader_train, dataloader_val, dataloader_test):
    epochs = 200
    model.train()

    # 6. Epochs
    for epoch in range(0, epochs):

        running_loss = 0.0
        model.train()

        # 7. Enumerate data
        for batch_id, data in enumerate(dataloader_train):

            img_fname, input, target = data

            # print("img_fname: ", img_fname)
            # print("input shape: ", input.size())
            # print("target shape: ", target.size())

            input = input.to(device='cuda')
            target = target.to(device='cuda')

            # 8. Model forward, backward, loss
            output = model.forward(input)
            # print("output shape: ", output.size())

            loss = criterio(output, target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() 

        total = batch_id + 1
        print("[Train] -- epoch: {} loss: {:.6f} ".format(epoch, running_loss/total))

        # Run test
        if (epoch + 1) % 10 == 0:
            test(model, dataloader_val, epoch)
            test(model, dataloader_test, epoch)

    print("End of training 0_0 ")

    model_fname = "TN_unet_supervised_vectra2ch_model_eroded_labels_focal_loss_flip.pt"
    torch.save(model, model_fname)

def get_image_name(fname):
    
    prefix = fname[0].split('/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/combined/')[-1].split('.tiff')[0]
    
    return prefix

# 6. Eval / Test function 
def test(model, dataloader_test, epoch, save=False):

    model.eval()

    running_auc = 0.0 
    running_auc = 0.0
    running_jacc = 0.0
    running_miou = 0.0 
    running_mdice = 0.0 
    running_ap = 0.0
    
    for batch_id, data in enumerate(dataloader_test):

        img_fname, input, target = data
        # print("fname : ", img_fname[0])

        input = input.to(device='cuda')
        target = target.to(device='cuda')

        # 8. Model forward, backward, loss
        output = model.forward(input)

        # print("predictions shape : ", output.shape) 
        
        if save == True:
            prefix = get_image_name(img_fname)
            save_pred_dir = '/nadeem_lab/Gunjan/code/impartial-lite/ImPartial/segmentation/predictions_focal_loss_flip/' 
            fpred = os.path.join(save_pred_dir, prefix + '.npy')
            
            np.save(fpred, output[0,:,:,:].cpu().detach().numpy()) 
            # (1) save predictions using fname_pred.npy 
            # save out to separate numpy file on disk 
            pass

        labels_gt, _ = ndimage.label(target.cpu().detach().numpy() > 0.5)

        metrics = get_performance(labels_gt, output.cpu().detach().numpy())
        # simplify this:
        running_auc += metrics['auc']
        running_jacc += metrics['Jacc'] 
        running_miou += metrics['mIoU'] 
        running_mdice += metrics['mDice'] 
        running_ap += metrics['AP'] 

    total = batch_id + 1 
    print("[Test]  -- epoch: {} AUC: {:.4f} Jacc: {:.4f} mIoU: {:.4f} mDice: {:.4f} AP: {:.4f}".format(epoch, running_auc/total, running_jacc/total, running_miou/total, running_mdice/total, running_ap/total))


# Add argparse code:


# convert this into command line argument 
# mode = 'train'
mode = 'test'

if mode == 'train':
    train(model, criterio, optimizer, dataloader_train, dataloader_val, dataloader_test)

# # #load model
if mode == 'test':
    model_file_path = '/nadeem_lab/Gunjan/code/impartial-lite/ImPartial/segmentation/TN_unet_supervised_vectra2ch_model_eroded_labels_focal_loss_flip.pt'
    model = torch.load(model_file_path)
    # model.load_state_dict(model_file_path)
    test(model, dataloader_test, 2, save=True)

#plot out
#commit notebooks to git. 
# test(model, dataloader_test)