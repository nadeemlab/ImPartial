import os
import logging
import argparse
import mlflow 
import configparser 
import multiprocessing 
import numpy as np 
import random

import torch
import torch.utils.data
from torchvision import transforms

from impartial.dataprocessing.datasets import ImageBlindSpotDataset, ImageSegDataset, ImageDataset
from impartial.dataprocessing.datasets import Normalize, ToTensor, RandomFlip, RandomRotate, RandomPermuteChannel, Resize, ResizeInfer
from impartial.dataprocessing.reader import DataProcessor, prepare_data_train, prepare_data_test, prepare_data_infer, plot_sample, plot_patch_sample
from impartial.general.training import Trainer
from impartial.general.networks import UNet
from impartial.general.losses import ImpartialLoss, LossConfig
from impartial.general.config import ImConfigIni, config_to_dict
from impartial.general.utils import model_params_load
from impartial.general.utils_mlflow import init_mlflow


def get_output_size(classification_tasks):
    output_size = 0
    for task in classification_tasks.values():
        ncomponents = np.sum(np.array(task['ncomponents']))
        output_size += ncomponents #components of the task

        nrec = len(task['rec_channels'])
        output_size += ncomponents * nrec

    return output_size

def plot_patches_sample(dataloader, output_dir):
    output_file_path_dir = os.path.join(output_dir, 'samples/patches')
    if not os.path.exists(output_file_path_dir):
        os.makedirs(output_file_path_dir)

    count_p = 0
    for _, data in enumerate(dataloader):

        x = data['input']
        mask = data['mask']
        scribble = data['scribble']
        target = data['target'] 
        print(x.shape, mask.shape, scribble.shape, target.shape)
        
        bs = x.size(0)
        for i in range(bs):
            patch_x = x[i]
            patch_mask = mask[i]
            patch_scribble = scribble[i]
            patch_target = target[i]

            patch_sample = {
                'input': patch_x,
                'mask': patch_mask,
                'scribble': patch_scribble,
                'target': patch_target
            }
            output_file_path = os.path.join(output_file_path_dir, 'ps_{}.png'.format(count_p))
            plot_patch_sample(patch_sample, output_file_path)    
            mlflow.log_artifact(output_file_path, "samples/patches")    

            count_p += 1

            if count_p >= 20:
                return


random.seed(71)
torch.manual_seed(51)
np.random.seed(13)

parser = argparse.ArgumentParser(description='Impartial Pipeline')
parser.add_argument('--experiment_name', required=True, help='experiment name')
parser.add_argument('--run_name', required=True, help='run name')
parser.add_argument('--output_dir', required=True, type=str, help='output_dir')
parser.add_argument('--log_file_name', required=True, type=str, help='log_file_name')
parser.add_argument('--config', type=str, required=True, help='config file')
parser.add_argument('--mode', type=str, default="train", help='train / eval')
parser.add_argument('--resume', required=False, type=str, default='', help='path to a pre-trained model')
parser.add_argument('--scribble_rate', required=False, type=float, default=1.0, help='scribble rate')
parser.add_argument('--train_sample', required=False, type=float, default=1.0, help='percentage images')

args = parser.parse_args()

print("initiate mlflow ... ")
init_mlflow(experiment_name=args.experiment_name, run_name=args.run_name)

# Log all input arguments as mlflow parameters
mlflow.log_params(vars(args))


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

os.makedirs(args.output_dir, exist_ok=True)
output_dir = args.output_dir

log_file_name = os.path.join(output_dir, args.log_file_name)
logging.basicConfig(filename=log_file_name, encoding='utf-8', level=logging.DEBUG)

# Setup training device 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

 
# 0. Configs:
mean = True
std = False
rec_channels = [int(x) for x in config.data.rec_channels.split(',')]
ncomponents = [int(x) for x in config.data.ncomponents.split(',')]
classification_tasks = {'0': {'classes': 1, 'rec_channels': rec_channels, 'ncomponents': ncomponents}}
print(classification_tasks)

weight_objectives = {'seg_fore': float(config.loss.seg_fore), 
                     'seg_back': float(config.loss.seg_back), 
                     'rec': float(config.loss.rec), 
                     'reg': float(config.loss.reg)}

print(weight_objectives)

for key in classification_tasks.keys():
    nrec = len(classification_tasks[key]['rec_channels'])
    nclasses = classification_tasks[key]['classes']

    if 'weight_classes' not in classification_tasks[key].keys():
        classification_tasks[key]['weight_classes'] = [1/nclasses for _ in range(nclasses)]
    if 'weight_rec_channels' not in classification_tasks[key].keys():
        classification_tasks[key]['weight_rec_channels'] = [1/nrec for _ in range(nrec)]

weight_tasks = None # per segmentation class reconstruction weight
if weight_tasks is None:
    weight_tasks = {}
    for key in classification_tasks.keys():
        weight_tasks[key] = 1/len(classification_tasks.keys())

mlflow.log_params(classification_tasks)

#transform 
augmentations = True
normstd = False # normalization

# 1. Data processor 
# train: image + gt 
# test: image + gt 
# infer: image 
dataProcessor = DataProcessor(config_data.dataset_dir, 
                              extension_image=config_data.extension_image, 
                              extension_label=config_data.extension_label)

train_paths, test_paths, infer_paths = dataProcessor.get_file_list()
print("train_paths: ", len(train_paths))

random.shuffle(train_paths)
n_val = int(len(train_paths) * 0.15)
val_paths = train_paths[:n_val]
train_paths = train_paths[n_val:]
print("train_paths: ", len(train_paths))
print("val_paths: ", len(val_paths))

# Fix it: in eval model we should directly jump to eval
if args.mode == "eval":
    train_paths = train_paths[:4]


print("train_paths: ", train_paths[:5])
print("test_paths: ", test_paths[:5])
print("infer_paths: ", infer_paths)

# sample trining images based on sampling rate: 
train_paths = train_paths[:int(len(train_paths) * args.train_sample)]
val_paths = val_paths[:int(len(val_paths) * args.train_sample)]
print("train_paths: ", len(train_paths))
print("val_paths: ", len(val_paths))
print("test_paths: ", len(test_paths))
print("infer_paths: ", len(infer_paths))


with multiprocessing.Pool() as pool:
    train_data = pool.starmap(prepare_data_train, [((image_path, roi_path), args.scribble_rate) for (image_path, roi_path) in train_paths])
    train_data = [d for d in train_data if d is not None]

with multiprocessing.Pool() as pool:
    val_data = pool.starmap(prepare_data_train, [((image_path, roi_path), args.scribble_rate) for (image_path, roi_path) in val_paths])
    val_data = [d for d in val_data if d is not None]

print("Train samples: save a few for visualization")
# only save a small sample
for sample in train_data[:20]:
    image_path = sample['name']
    output_file_path = os.path.join(output_dir, image_path.split('/')[-1] + '.png')
    plot_sample(sample, output_file_path)
    mlflow.log_artifact(output_file_path, "samples/train")
    
print("Val samples: save a few for visualization")
for sample in val_data[:5]:
    image_path = sample['name']
    output_file_path = os.path.join(output_dir, image_path.split('/')[-1] + '.png')
    plot_sample(sample, output_file_path)
    mlflow.log_artifact(output_file_path, "samples/val")

print("Test samples")
test_data = []
for image_path, roi_path in test_paths: 
    sample = prepare_data_test((image_path, roi_path))
    test_data.append(sample)


infer_data = []
for image_path in infer_paths: 
    sample = prepare_data_infer(image_path)
    infer_data.append(sample)


mlflow.log_param("num_train_paths", len(train_paths))
mlflow.log_param("num_test_paths", len(test_paths))
mlflow.log_param("num_infer_paths", len(infer_paths))


patch_size = [int(x) for x in config_train.patch_size.split(',')]
npatch_image = [int(x) for x in config_train.num_patches_per_image.split(',')]
patches_dict = {ps: npatch for ps, npatch in zip(patch_size, npatch_image)}
train_img_size = int(config_train.train_img_size)
infer_img_size = int(config_train.infer_img_size)


transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
if augmentations:
    transforms_list.append(RandomFlip())
    transforms_list.append(RandomRotate())
    # transforms_list.append(RandomPermuteChannel())
    
# transforms_list.append(Resize(size=(train_img_size, train_img_size)))
transforms_list.append(ToTensor(dim_data=3))
transform_train = transforms.Compose(transforms_list)


dataset_train = ImageBlindSpotDataset(train_data, transform=transform_train, patches_dict=patches_dict, train_img_size=train_img_size,)
dataset_train.sample_patches_data()
print("train_dataset.data_list: ", len(dataset_train))


transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
# transforms_list.append(Resize(size=(train_img_size, train_img_size)))
transforms_list.append(ToTensor(dim_data=3))
transform_val = transforms.Compose(transforms_list)

dataset_val = ImageBlindSpotDataset(val_data, transform=transform_val, patches_dict=patches_dict, train_img_size=train_img_size)
dataset_val.sample_patches_data()
print("val_dataset.data_list: ", len(dataset_val))


transforms_list = []
if normstd:
    transforms_list.append(Normalize(mean=0.5, std=0.5))
transforms_list.append(ResizeInfer(size=(infer_img_size, infer_img_size)))
transforms_list.append(ToTensor())
transform_test = transforms.Compose(transforms_list)


# Full image inference
dataset_eval_test = ImageSegDataset(test_data, transform=transform_test) # with labels
dataset_infer = ImageDataset(test_data, transform=transform_test) # without labels 
 
print("train_dataset: ", len(dataset_train))
print("val_dataset: ", len(dataset_val))

mlflow.log_param("train_dataset", len(dataset_train))
mlflow.log_param("val_dataset", len(dataset_val))


# 1.1 Create dataloaders from dataset 
# create transforms, dataloader 

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=int(config_train.batch_size), shuffle=True, num_workers=int(config_train.num_workers))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=int(config_train.batch_size), shuffle=True, num_workers=int(config_train.num_workers))

dataloader_eval_test = torch.utils.data.DataLoader(dataset_eval_test, batch_size=1, shuffle=False, num_workers=4) 
dataloader_infer = torch.utils.data.DataLoader(dataset_infer, batch_size=1, shuffle=False, num_workers=4) 


print("Plotting patches sample ...")
plot_patches_sample(dataloader_train, output_dir)


# 2. model, optimizer, loss function
n_channels = int(config_data.n_channels)
n_output = int(config_data.n_output)
n_output = get_output_size(classification_tasks)

drop_encoder_decoder = False
drop_last_conv = False
p_drop = 0.5
## Check this again
## https://github.com/nadeemlab/ImPartial/blob/main/impartial/Impartial_classes.py#L138 

if int(config_train.mcdrop_it) > 0:
    drop_encoder_decoder = True
    drop_last_conv = False


if config_train.model_name == "unet":
    model = UNet(    
        n_channels,
        n_output, # # this is actuall n_output ---> ONLY FOR: if mean = True, len(rec) = 2, class = 1, ncomponents = [2,2] 
        depth=int(config.model.depth),
        base=int(config.model.base),
        activation='relu',
        batchnorm=False,
        dropout=drop_encoder_decoder, # TODO: check this
        dropout_lastconv=drop_last_conv,
        p_drop=p_drop)

model = model.to(device)
# model = torch.nn.DataParallel(model)

if config_train.optim_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=float(config_train.lr), 
                                 weight_decay=float(config_train.optim_weight_decay))
elif config_train.optim_name == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config_train.lr),
                                  weight_decay=float(config_train.optim_weight_decay))
else:
    raise ValueError(f"Optimizer {config_train.optim_name} not supported")

# Learning rate scheduler - ReduceLROnPlateau (recommended for UNet/segmentation)
# Reduces LR when validation loss plateaus, more adaptive than fixed schedules
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.8, patience=10, 
#     min_lr=0.0000001, verbose=True
# )

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# warmup = LinearLR(optimizer, start_factor=0.1, total_iters=20)
# cosine = CosineAnnealingLR(optimizer, T_max=int(config_train.epochs) - 5, eta_min=1e-6)
# scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])


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
                  scheduler=scheduler,
                  output_dir=output_dir,
                  n_output=n_output,
                  epochs=int(config_train.epochs),
                  mcdrop_it=int(config_train.mcdrop_it),
                  eval_freq=int(config_train.eval_freq),
                  eval_sample_freq=int(config_train.eval_sample_freq)
                )

if args.mode == "train":
    trainer.train(
        dataloader_train=dataloader_train, 
        dataloader_val=dataloader_val, 
        dataloader_eval=dataloader_eval_test, 
        dataloader_infer=dataloader_infer
        ) 


if args.mode == "eval":
    print(len(dataset_eval_test))
    trainer.eval_freq = 1
    trainer.evaluate(
        dataloader_eval=dataloader_eval_test, 
        is_save=True,
        dilate=True,
        threshold=0.7,
        n_iter=2
        ) 

if args.mode == "infer":
    print(len(dataset_eval_test))
    trainer.eval_freq = 1
    trainer.eval_sample_freq = 1
    trainer.evaluate(
        dataloader_eval=dataloader_eval_test, 
        is_save=True,
        dilate=True,
        threshold=0.7,
        n_iter=2
        ) 

mlflow.end_run()
