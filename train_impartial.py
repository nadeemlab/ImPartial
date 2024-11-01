import os
import logging
import argparse
import mlflow 
import configparser 
import multiprocessing 

import torch
import torch.utils.data
from torchvision import transforms

from dataprocessing.datasets import ImageBlindSpotDataset, ImageSegDataset, ImageDataset, Normalize, ToTensor, RandomFlip
from dataprocessing.reader import prepare_data_train, prepare_data_test, prepare_data_infer, DataProcessor, save_model, plot_sample, plot_patch_sample
from general.training import Trainer
from general.networks import UNet
from general.losses import ImpartialLoss, LossConfig
from general.config import ImConfigIni, config_to_dict
from general.utils import model_params_load


def init_mlflow(experiment_name, run_name):
    # mlflow.set_tracking_uri("file:///nadeem_lab/Gunjan/mlruns")
    # mlflow.set_tracking_uri("http://10.0.3.10:8000") # old
    mlflow.set_tracking_uri("http://10.0.3.12:8000")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.start_run(run_name=run_name)


parser = argparse.ArgumentParser(description='Impartial Pipeline')
parser.add_argument('--experiment_name', required=True, help='experiment name')
parser.add_argument('--run_name', required=True, help='run name')
parser.add_argument('--output_dir', required=True, type=str, help='output_dir')
parser.add_argument('--log_file_name', required=True, type=str, help='log_file_name')
parser.add_argument('--scribble_rate', required=False, type=float, default=1.0, help='scribble rate')
parser.add_argument('--config', type=str, required=True, help='config file')
parser.add_argument('--mode', type=str, default="train", help='train / eval')
parser.add_argument('--resume', required=False, type=str, default='', help='path to a pre-trained model')

args = parser.parse_args()

print("initiate mlflow ... ")
init_mlflow(experiment_name=args.experiment_name, run_name=args.run_name)

parser = configparser.ConfigParser()
parser.read_file(open(args.config))
config = ImConfigIni(parser)

config_train = config.train 
config_data = config.data 
config_loss = config.loss  # unused
config_model = config.model

print('train:', config_train.epochs, config_train.lr, config_train.mcdrop_it)
print('data:', config_data.n_channels, config_data.n_output)
print('data:', config_data.dataset_dir, config_data.extension_image, config_data.extension_label)

print('config_data:', config_data)
print('config_model:', config_loss)
print('config_unet:', config_model)


config_dict = config_to_dict(args.config)
print(config_dict)
mlflow.log_params(config_dict)

print("initialize logging ... ")
logger = logging.getLogger(__name__)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    print("Creating output dir ...")
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
mean = True
std = False
rec_channels = [int(x) for x in config.data.rec_channels.split(',')]
ncomponents = [int(x) for x in config.data.ncomponents.split(',')]
classification_tasks = {'0': {'classes': 1, 'rec_channels': rec_channels, 'ncomponents': ncomponents}}
print(classification_tasks)

# classification_tasks = {'0': {'classes': 1, 'rec_channels': [0,1,2], 'ncomponents': [2,2]}}  # list containing classes 'object types'
# classification_tasks = {'0': {'classes': 1, 'rec_channels': [0,1], 'ncomponents': [2,2]}}  # list containing classes 'object types'
# classification_tasks = {'0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2,2]}}  # list containing classes 'object types'
weight_tasks = None # per segmentation class reconstruction weight
weight_objectives = {'seg_fore': float(config.loss.seg_fore), 
                     'seg_back': float(config.loss.seg_back), 
                     'rec': float(config.loss.rec), 
                     'rec': float(config.loss.rec)}

print(weight_objectives)
# weight_objectives = {'seg_fore': 0.45, 'seg_back': 0.45, 'rec': 0.1, 'reg': 0.0}
# weight_objectives = {'seg_fore': 0.4, 'seg_back': 0.4, 'rec': 0.2, 'reg': 0.0}

# exit()

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

# train: image + gt 
# test: image + gt 
# infer: image 
dataProcessor = DataProcessor(config_data.dataset_dir, extension_image=config_data.extension_image, extension_label=config_data.extension_label)
train_paths, test_paths, infer_paths = dataProcessor.get_file_list()

print("train_paths: ", train_paths[:5])
print("test_paths: ", test_paths[:5])
print("infer_paths: ", infer_paths)

# train_paths = train_paths[:100]
# test_paths = test_paths[:50]
test_paths = test_paths


print("train_paths: ", len(train_paths))
print("test_paths: ", len(test_paths))
print("infer_paths: ", len(infer_paths))


with multiprocessing.Pool() as pool:
    train_data = pool.starmap(prepare_data_train, [((image_path, roi_path), args.scribble_rate) for (image_path, roi_path) in train_paths])
    train_data = [d for d in train_data if d is not None]

print("Train samples: save a few for visualization")
# only save a small sample
for sample in train_data[:20]:
    image_path = sample['name']
    output_file_path = os.path.join(output_dir, image_path.split('/')[-1] + '.png')
    plot_sample(sample, output_file_path)
    mlflow.log_artifact(output_file_path, "samples")
    train_data.append(sample)


test_data = []
for image_path, roi_path in test_paths: 
    sample = prepare_data_test((image_path, roi_path))
    # output_file_path = os.path.join(output_dir, image_path.split('/')[-1] + '.png')
    # plot_sample(sample, output_file_path)
    # mlflow.log_artifact(output_file_path, "samples")
    test_data.append(sample)


infer_data = []
for image_path in infer_paths: 
    sample = prepare_data_infer(image_path)
    infer_data.append(sample)


transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
if augmentations:
    transforms_list.append(RandomFlip())
transforms_list.append(ToTensor(dim_data=3))
transform_train = transforms.Compose(transforms_list)


npatch_image_train = int(config_train.num_patches_per_image)
npatch_image_val = int(0.2*npatch_image_train)
train_dataset = ImageBlindSpotDataset(train_data, transform=transform_train, npatch_image=npatch_image_train)
print("Train: Create Patches ...")
train_dataset.sample_patches_data()
print("train_dataset.data_list: ", len(train_dataset.data_list))

transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
transforms_list.append(ToTensor())
transform_val = transforms.Compose(transforms_list)

val_dataset = ImageBlindSpotDataset(train_data, transform=transform_val, validation=True, npatch_image=npatch_image_val)
print("Val: Create Patches ...")
val_dataset.sample_patches_data()
print("val_dataset.data_list: ", len(val_dataset.data_list))

# Full image inference
dataset_eval_train = ImageSegDataset(train_data, transform=transform_val) # with labels
dataset_eval_test = ImageSegDataset(test_data, transform=transform_val) # with labels

dataset_infer = ImageDataset(test_data, transform=transform_val) # without labels 
 
print("train_dataset: ", len(train_dataset))
print("val_dataset: ", len(val_dataset))

# 1.1 Create dataloaders from dataset 
# create transforms, dataloader 

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=int(config_train.batch_size), shuffle=True, num_workers=int(config_train.num_workers))
dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=int(config_train.batch_size), shuffle=False, num_workers=int(config_train.num_workers))

dataloader_eval_train = torch.utils.data.DataLoader(dataset_eval_train, batch_size=1, shuffle=False, num_workers=20) 
dataloader_eval_test = torch.utils.data.DataLoader(dataset_eval_test, batch_size=1, shuffle=False, num_workers=20) 
dataloader_infer = torch.utils.data.DataLoader(dataset_infer, batch_size=1, shuffle=False, num_workers=20) 

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


if args.resume != '' and os.path.exists(args.resume):
    print("Loading pre-trained model: ", args.resume)
    model_params_load(args.resume, model, optimizer, device)
else:
    print("Loading args.resume does not exists or invalid: ", args.resume)

# 3. Training code 
trainer = Trainer(device=device, 
                  classification_tasks=classification_tasks, 
                  model=model, criterion=criterion, optimizer=optimizer, 
                  output_dir=output_dir,
                  n_output=n_output,
                  epochs=int(config_train.epochs),
                  mcdrop_it=int(config_train.mcdrop_it))

if args.mode == "train":
    trainer.train(
        dataloader_train=dataloader_train, 
        dataloader_val=dataloader_val, 
        dataloader_eval=dataloader_eval_test, 
        dataloader_infer=dataloader_infer
        ) 


if args.mode == "eval":
    trainer.evaluate(
        dataloader_eval=dataloader_eval_test, 
        ) 



# path = os.path.join(output_dir, 'best.pth')
# save_model(model, path)


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