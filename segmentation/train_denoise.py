import torch
from scipy import ndimage
import sys

from torchvision import transforms

sys.path.append('../')

from general.networks import UNet
from segmentation.dataset import ImageDenoiseDatasetTiff
from segmentation.datalist import train_data, test_data

# 1. Model
model = UNet(
    n_channels=2,
    n_classes=2,
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

transform_train = []
transform_train.append(transforms.ToTensor())
transform_train = transforms.Compose(transform_train)

transform_test = []
transform_test.append(transforms.ToTensor())
transform_test = transforms.Compose(transform_test)

# # 3.1 dataset and dataloaders
# dataset_files = {} 
# dataset_files['images'] = image_list
# dataset_files['labels'] = label_list

image_list, label_list = train_data(data_dir)
dataset_train = ImageDenoiseDatasetTiff(data_dir, image_list, transform_train)


# (2) Make dataset test 
# dataset_test = CPTissuenetDatasetTiff(data_dir, transform_test)
image_list, label_list = test_data(data_dir)
dataset_test = ImageDenoiseDatasetTiff(data_dir, image_list, transform_test)

print("Size dataset_train: ", len(dataset_train))
print("Size dataset_test: ", len(dataset_test))

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=16)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=8)


# 4. Loss Function
criterio = torch.nn.MSELoss()


# 5. Train function 
def train(model, criterio, optimizer, dataloader_train, dataloader_test):
    epochs = 50
    model.train()

    # 6. Epochs
    for epoch in range(0, epochs):

        running_loss = 0.0

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

        print("[Train] -- epoch: {} loss: {:.6f} ".format(epoch, running_loss/batch_id))

        # Run test
        if (epoch + 1) % 5 == 0:
            test(model, dataloader_test, epoch)

    print("End of training 0_0 ")

    model_fname = "TN_unet_mse_vectra2ch_model.pt"
    torch.save(model, model_fname)


# 6. Eval / Test function 
def test(model, dataloader_test, epoch, save=False):

    running_mse = 0.0 
    for batch_id, data in enumerate(dataloader_test):

        img_fname, input, target = data

        input = input.to(device='cuda')
        target = target.to(device='cuda')

        # 8. Model forward, backward, loss
        output = model.forward(input)
        
        loss = criterio(output, target)

        # print("predictions : ", output.shape) 
        
        if save == True:
            # (1) save predictions using fname_pred.npy 
            # save out to separate numpy file on disk 
            pass

        running_mse += loss.item()

    print("[Test]  -- epoch: {} mse: {:.4f}".format(epoch, running_mse/batch_id))




train(model, criterio, optimizer, dataloader_train, dataloader_test)
# test(model, dataloader_test)

# #load model
# model_file_path = '/nadeem_lab/Gunjan/code/impartial-lite/ImPartial/models/TN_unet_mse_vectra2ch_model.pt'

# model = torch.load(model_file_path, map_location=torch.device('cpu'))
# # model.load_state_dict(model_pth)

#plot out
#commit notebooks to git. 
# test(model, dataloader_test)