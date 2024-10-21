import os
import logging
import argparse
import mlflow 

import torch
from torchvision import transforms

import torch
import torch.utils.data
from torchvision import transforms

from dataprocessing.datasets import ImageBlindSpotDataset, ImageSegDataset, ImageDataset, Normalize, ToTensor, RandomFlip
from dataprocessing.reader import prepare_data, prepare_data_test, DataProcessor, save_model, plot_sample, plot_patch_sample
from general.training import Trainer
from general.networks import UNet
from general.losses import ImpartialLoss, LossConfig

from general.config import ImConfigIni, config_to_dict
import configparser 


def init_mlflow(experiment_name, run_name):
    # mlflow.set_tracking_uri("file:///nadeem_lab/Gunjan/mlruns")
    mlflow.set_tracking_uri("http://10.0.3.10:8000")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.start_run(run_name=run_name)



parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--experiment_name', required=True, help='experiment name')
parser.add_argument('--run_name', required=True, help='run name')
parser.add_argument('--output_dir', required=True, type=str, help='output_dir')
parser.add_argument('--log_file_name', required=True, type=str, help='log_file_name')
parser.add_argument('--scribble_rate', required=False, type=float, default=1.0, help='scribble rate')

args = parser.parse_args()


print("initiate mlflow ... ")
init_mlflow(experiment_name=args.experiment_name, run_name=args.run_name)


##### 

# config_filename = 'config/default.ini'
config_filename = 'config/vectra_2ch.ini'

parser = configparser.ConfigParser()
parser.read_file(open(config_filename))
config = ImConfigIni(parser)

config_train = config.train 
config_data = config.data 
config_loss = config.loss  # unused
config_model = config.model

print('train:', config_train.epochs, config_train.lr, config_train.mcdrop_it)
print('data:', config_data.n_channels, config_data.n_output)
print('data:', config_data.dataset_dir, config_data.extension)

print('config_data:', config_data)
print('config_model:', config_loss)
print('config_unet:', config_model)


config_dict = config_to_dict(config_filename)
print(config_dict)
mlflow.log_params(config_dict)

# exit()

#####



print("initialize logging ... ")

logger = logging.getLogger(__name__)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    print("creating output dir")
    os.makedirs(output_dir)

log_file_name = os.path.join(output_dir, args.log_file_name)
logging.basicConfig(filename=log_file_name, encoding='utf-8', level=logging.DEBUG)

# Setup training device 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


"""
1. Read dataset - images + zip
2. Create dataset ? 
3. Create model, optimizer, loss function 
4. Training code / loop 
5. Evaluation ? 
6. Save results / visualize 

"""

# 0. Configs:
# classification_tasks = {'0': {'classes': 1, 'rec_channels': [0,1,2], 'ncomponents': [2,2]}}  # list containing classes 'object types'
classification_tasks = {'0': {'classes': 1, 'rec_channels': [0,1], 'ncomponents': [2,2]}}  # list containing classes 'object types'
# classification_tasks = {'0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2,2]}}  # list containing classes 'object types'
mean = True
std = False
weight_tasks = None #per segmentation class reconstruction weight
weight_objectives = {'seg_fore': 0.45, 'seg_back': 0.45, 'rec': 0.1, 'reg': 0.0}

### Checkpoint ensembles
nsaves = 1 # number of checkpoints ensembles
val_model_saves_list = []
train_model_saves_list = []
reset_optim = True
reset_validation = False

save_intermediates = False

for key in classification_tasks.keys():
    nrec = len(classification_tasks[key]['rec_channels'])
    nclasses = classification_tasks[key]['classes']

    if 'weight_classes' not in classification_tasks[key].keys():
        classification_tasks[key]['weight_classes'] = [1/nclasses for _ in range(nclasses)]
    if 'weight_rec_channels' not in classification_tasks[key].keys():
        classification_tasks[key]['weight_rec_channels'] = [1/nrec for _ in range(nrec)]

if weight_tasks is None:
    weight_tasks = {}
    for key in classification_tasks.keys():
        weight_tasks[key] = 1/len(classification_tasks.keys())


mlflow.log_params(classification_tasks)

#transform 
augmentations = True
normstd = False #normalization
     
# 1. Data processor 

# dataset_dir = '/nadeem_lab/Gunjan/data/Vectra_WC_2CH_tiff_full_labels/'
# dataset_dir = '/nadeem_lab/Gunjan/data/DAPI1CH_png/'
# dataset_dir = '/nadeem_lab/Gunjan/data/cellpose_tiff_sample/'
# dataset_dir = '/nadeem_lab/Gunjan/data/Vectra_WC_2CH_tiff_batchsize64/' ## has only partial scribbles
# dataset_dir = '/nadeem_lab/Gunjan/data/segpath_cd3_sample/'
# dataset_dir = '/nadeem_lab/Gunjan/data/impartial_segpath_sample/set/'
# dataset_dir = '/nadeem_lab/Gunjan/data/impartial_segpath_sample/set2/'
# dataset_dir = '/nadeem_lab/Gunjan/data/impartial_segpath_sample/set4/'
# dataset_dir = '/nadeem_lab/Gunjan/data/cpdmi_3ch_full_label_dataset/'

dataProcessor = DataProcessor(config_data.dataset_dir, extension=config_data.extension)
train_paths, test_paths = dataProcessor.get_file_list()
print("train_paths: ", train_paths )
print("test_paths: ", test_paths )


train_data = []
for image_path, roi_path in train_paths: 
    sample = prepare_data(image_path, roi_path, scribble_rate=args.scribble_rate)
    output_file_path = os.path.join(output_dir, image_path.split('/')[-1] + '.png')
    plot_sample(sample, output_file_path)
    mlflow.log_artifact(output_file_path, "samples")
    train_data.append(sample)

test_data = []
for image_path in test_paths: 
    sample = prepare_data_test(image_path)
    test_data.append(sample)


transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
if augmentations:
    transforms_list.append(RandomFlip())
transforms_list.append(ToTensor(dim_data=3))

transform_train = transforms.Compose(transforms_list)

#npatches =1000
train_dataset = ImageBlindSpotDataset(train_data, transform=transform_train, npatch_image=1000)
print("train_dataset.data_list: ", len(train_dataset.data_list))
train_dataset.sample_patches_data()

transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
transforms_list.append(ToTensor())
transform_val = transforms.Compose(transforms_list)

val_dataset = ImageBlindSpotDataset(train_data, transform=transform_val, validation=True, npatch_image=500)
val_dataset.sample_patches_data()

# Full image inference
dataset_eval = ImageSegDataset(train_data, transform=transform_val) # with labels
dataset_infer = ImageDataset(test_data, transform=transform_val) # without labels 
 
print("train_dataset: ", len(train_dataset))
print("val_dataset: ", len(val_dataset))

# 1.1 Create dataloaders from dataset 
# create transforms, dataloader 

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=int(config_train.batch_size), shuffle=True, num_workers=int(config_train.num_workers))
dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=int(config_train.batch_size), shuffle=False, num_workers=int(config_train.num_workers))
dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=2) 
dataloader_infer = torch.utils.data.DataLoader(dataset_infer, batch_size=1, shuffle=False, num_workers=2) 

"""
# Code to plot paches
for idx, patch_sample in enumerate(train_dataset):
    output_file_path_dir = os.path.join(output_dir, 'patch_samples')
    if not os.path.exists(output_file_path_dir):
        os.makedirs(output_file_path_dir)

    output_file_path = os.path.join(output_file_path_dir, 'ps_{}.png'.format(idx))
    print(output_file_path)
    plot_patch_sample(patch_sample, output_file_path)    
    mlflow.log_artifact(output_file_path, "patches")    
"""


# 2. model, optimizer, loss function
# change
n_channels = int(config_data.n_channels)
n_output = int(config_data.n_output)

# n_output = 16
# n_output = 12 
# n_channels = 2
# n_output = 8
# n_channels = 1


drop_encoder_decoder = False
drop_last_conv = False
p_drop = 0.2 #0.5
## Check this again
## https://github.com/nadeemlab/ImPartial/blob/main/impartial/Impartial_classes.py#L138 

if int(config_train.mcdrop_it) > 0:
    drop_encoder_decoder = True
    drop_last_conv = False

model = UNet(    
    n_channels,
    n_output, # # this is actuall n_output ---> ONLY FOR: if mean = True, len(rec) = 2, class = 1, ncomponents = [2,2] 
    depth=4,
    base=64,
    activation='relu',
    batchnorm=False,
    dropout=drop_encoder_decoder, # TODO: check this
    dropout_lastconv=drop_last_conv,
    p_drop=p_drop)


model = model.to(device)
# model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=float(config_train.lr), weight_decay=float(config_train.optim_weight_decay))

# criterion 
# ------------------------- losses --------------------------------#

config_loss = LossConfig(device=device, 
                         classification_tasks=classification_tasks, 
                         weight_tasks=weight_tasks, 
                         weight_objectives=weight_objectives)

criterion = ImpartialLoss(loss_config=config_loss)

# 3. Training code 
trainer = Trainer(device=device, 
                  classification_tasks=classification_tasks, 
                  model=model, criterion=criterion, optimizer=optimizer, 
                  output_dir=output_dir,
                  n_output=n_output,
                  epochs=int(config_train.epochs),
                  mcdrop_it=int(config_train.mcdrop_it))

trainer.train(dataloader_train=dataloader_train, 
              dataloader_val=dataloader_val, 
              dataloader_eval=dataloader_eval, 
              dataloader_infer=dataloader_infer) 

path = os.path.join(output_dir, 'best.pth')
save_model(model, path)


mlflow.end_run()


"""
Steps for ensemble using dropout:
Training:
- During training make sure dropout is enabled so that the model has dropout layer 
https://github.com/nadeemlab/ImPartial/blob/main/general/networks.py#L159 

Inference:
https://github.com/nadeemlab/ImPartial/blame/main/general/training.py#L107
https://github.com/nadeemlab/ImPartial/blob/main/general/training.py#L345 

- get mean & variance from ensemble
https://github.com/nadeemlab/ImPartial/blob/main/impartial/Impartial_functions.py#L329 


"""