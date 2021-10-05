<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./images/impartial-logo.png" width="50%">
    <h3 align="center"><strong>ImPartial: Interactive deep learning whole-cell segmentation using partial annotations</strong></h3>
    <p align="center">
    <a href="https://doi.org/10.1101/2021.01.20.427458">Read Link</a>
    |
    <a href="#colabnotebook">Google CoLab Demo</a>
    |
    <a href="#docker-file">Docker</a>
    |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Report Bug</a>
    |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Request Feature</a>
  </p>
</p>



## Prerequisites:
```
NVIDIA GPU (Tested on NVIDIA GPU)
CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification)
pandas>=1.2.4
numpy>=1.20.1
torchvision>=0.2.1
torch>=0.5.0
scikit-image>=0.18.1
scikit-learn>=0.24.1
scipy>=1.6.2
```

## Getting Started with ImPartial


## Terminology
```

pd_files
pd_files_scribbles
files_scribbles
classification tasks
ncomponents 
nclasses
rec_channels
```

## Datasets

| Dataset  | Description |
| ------------- | ------------- |
| `Deepcell`  | https://datasets.deepcell.org/  |
| `Cellpose`  | https://www.cellpose.org/  |

## Data Preparation and Pre-processing

### Steps
* Create dataset for ImPartial training
* Select training images split into train, val, test sets
* Use preprocessing files to generate input .npz files
* If Ground-truth labels available:
  * Use automated skeltonization on groud truth labels to generate scribbles .npz files
* Without Ground-truth labels:
  * Create scribbles using deepcell label (link) or other annotation tools 

  
### Code

There are two notebooks for preparing and processing a dataset. 

* Prepare_dataset.ipynb
	1. Load the dataset
	2. Select the images you want to use (save_list)
	3. Select the images from (save_list) to be used in training (train_list). Remaining images to be used in test.
	4. Set (savedir) path to save .npz files.
	5. Create `files.csv` which contains names (prefix) and path (input_dir) of train and test images (group)
  

* Preprocessing.ipynb (Automated scribble generation of training image from ground-truth labels)
	1. Read files.csv
	2. Get number of instances per segmentation class
	3. Define number of scribbles for labels
	4. Defines percentage validation region (val_perc = 0.4)
	5. Scribble .npz file is saved in the input_dir path defined in pd_files (files.csv)
	6. Scribble .csv file is also saved (contains parameters related to scribble)



## Training 

Create the following dataset specific training configuration in main_impartial.py:
```
n_channels = number of input image channels
classification_tasks = a python dict of tasks and corresponding number of classes, recunstruction channels
```

```python
    if cparser.dataset == 'Deepcell':
        scribble_fname = 'files_2tasks_10images_scribble_train_' + cparser.scribbles + '.csv'
        files_scribbles = os.path.join(data_dir, scribble_fname)
        pd_files_scribbles = pd.read_csv(files_scribbles) #scribbles

        n_channels = 2
        classification_tasks = {'0': {'classes': 1, 'rec_channels': [0,1], 'ncomponents': [2, 2]},
                                '1': {'classes': 1, 'rec_channels': [0], 'ncomponents': [1, 2]}}
```

	
* Select/ adjust training parameters 
  * Set input file paths in (impartial_bash.sh) file
  ```
    data_dir = "path to data directory containg .npz files"
	data_dir = '/nadeem_lab/Gunjan/data/impartial/' # example
  ```
  * Set output file paths in (impartial_bash.sh) file
  ```
    basedir_root = " path to output files "
	basedir_root = "/nadeem_lab/Gunjan/experiments/deepcell/models/" # example
  ```
  * Set dataset name
  * Set mcdropout, checkpoint ensembles, no. of epochs etc. 
	
	
* Training output is represented as .pickle file
  ```
  - output is a dictionary with tasks as keys (in case of 2 tasks [0,1], in case of 1 task [0])
  - each output[task] is a dictionary with:
	- 'class_segmentation': np.array size = (batch x nclasses x h x w) 
	- 'class_segmentation_variance': np.array size = (batch x nclasses x h x w)
  ```


Example training command
```
CUDA_VISIBLE_DEVICES=0 python3.8 main_impartial.py 
				--basedir=$basedir_root"Deepcell/s400/Impartial/" 
				--dataset="Deepcell" 
				--model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42" 
				--saveout=True 
				--scribbles=400 
				--gpu=0 
				--optim_regw=0.0001 
				--optim="adam" 
				--lr=0.0005 
				--gradclip=0 
				--seed=42 
				--train=True 
				--udepth="4" 
				--ubase="64" 
				--activation="relu" 
				--batchnorm=False 
				--seg_loss="CE" 
				--rec_loss="gaussian" 
				--nsaves=5 
				--mcdrop=True 
				--reset_optim=True 
				--reset_validation=False 
				--wfore=0.45 
				--wback=0.45 
				--wrec=0.1 
				--wreg=0.0 
				--ratio=0.95 
				--epochs=300 
				--batch=64 
				--load=False 
				
				> "output/path/to/logs"
```

## Evaluation using Pretrained model

To test the model use the following sample command. 
Modify the basedir, dataset, model_name to test a different model. 
Sample Premodels can be downloaded here.

Example evalualtion command
```
CUDA_VISIBLE_DEVICES=0 python3.8 main_impartial.py \
				--basedir=$basedir_root"Deepcell/s400/Impartial/" \
				--dataset="Deepcell" \
				--model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42" \
				--saveout=True \
				--scribbles=400 \
				--gpu=0 \
				--optim_regw=0.0001 \
				--optim="adam" \
				--lr=0.0005 \
				--gradclip=0 \
				--seed=42 \
				--train=False \
				--udepth="4" \
				--ubase="64" \
				--activation="relu" \
				--batchnorm=False \
				--seg_loss="CE" \
				--rec_loss="gaussian" \
				--nsaves=5 \
				--mcdrop=True \
				--reset_optim=True \
				--reset_validation=False  \
				--wfore=0.45 \
				--wback=0.45 \
				--wrec=0.1 \
				--wreg=0.0 \
				--ratio=0.95  \
				--epochs=300 --batch=64 \
				--load=True 

				> "output/path/to/logs"

```


## Demo with DeepCell 

This is a proof of concept demo of integration of ImPartial with DeepCell-Label for doing interactive deep learning whole-cell segmentation using partial annotations. 
Here you see the results after every few epochs during training of ImPartial on Tissuenet dataset.

![demo_nucleiseg_gif](./images/deepcell-label-nucleiSeg-image.gif)**Figure1**. *Nuclie segmentation.* The nuclie in input sample is give a few foreground(white) and background(red) scribbles. Image shows intermediate results after every 10th epoch. Final predictons are overlayed on ground truth.


![demo_cytoplasm_gif](./images/deepcell-label-cytoplasmSeg-image.gif)**Figure2**. *Cytoplasm segmentation.* The cytoplasm in input sample is give a few foreground(white) and background(red) scribbles. Image shows intermediate results after every 10th epoch. Final predictons are overlayed on ground truth.

## Google CoLab:
If you don't have access to GPU or appropriate hardware, we have also created [Google CoLab project](link) [To-Do] for your convenience. 
Please follow the steps in the provided notebook to install the requirements and run the training and testing scripts.
All the libraries and pretrained models have already been set up there. 
The user can directly run ImPartial on their dataset using the instructions given in the Google CoLab project. 


## Issues
Please report all issues on the public forum.


## License
© [Nadeem Lab](https://nadeemlab.org/) - ImPartial code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 



## Acknowledgments
[To-Do]


## Reference
If you find our work useful in your research or if you use parts of this code, please cite our paper:
```
[To-Do]
```



## Data Folders:

* /MIBI_2CH : 4  2-channel mibi images, segmentation classes: cytoplasm, nuclei in cytoplasm, nuclei out of cytoplasm
* /Vectra_2CH: 8  2-channel Vectra images, segmentation classes: cytoplasm, nuclei in cytoplasm, nuclei out of cytoplasm

In each dataset folder make sure to change the ‘input_dir’ column values in the .csv files for the correct path of the folder.
(e.g., 

data_dir = ‘/Data/MIBI_2CH/‘

pd_file = pd.read_csv(data_dir+'files.csv',index_col=0)

pd_file['input_dir'] = data_dir

pd_file.to_csv(data_dir + 'files.csv')
)




## Examples for running training baseline models :

#### Dataset MIBI_2CH with 150 scribbles:

- Impartial Model : file is /Impartial/main_impartial.py -> Here change the directory of the dataset files (line ~90)

```
python main_impartial.py --basedir="/data/natalia/models/MIBI2CH/s150/Impartial/" --dataset="MIBI2CH" --model_name="Im_2tasks_base64depth4relu_adam5e4_nsave6_segCEGauss_w04501_seed42" --scribbles=150 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=6 --reset_optim=True  --wfore=0.45 --wback=0.45 --wrec=0.1 --ratio=0.95  --epochs=400 --batch=64 --load=False > MIBI2CH_Im_2tasks_verbose.txt
```


- MumfordShah Model : file is /MumfordShah/main_ms.py -> Here change the directory of the dataset files 

```
python main_ms.py --basedir="/data/natalia/models/MIBI2CH/s150/MS/" --dataset="MIBI2CH" --model_name="MS_2tasks_base64depth4relu_adam5e4_nsave6_segCErecL2_w04501_seed42" --saveout=True --scribbles=150 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="L2" --nsaves=6 --reset_optim=True --wfore=0.45 --wback=0.45 --wrec=0.09 --wreg=0.01 --epochs=400 --batch=64 --load=False > MIBI2CH_MS_2tasks_verbose.txt
```

- Baseline Model : file is /Baseline/main_bs.py -> Here change the directory of the dataset files 

```
python main_bs.py --basedir="/data/natalia/models/MIBI2CH/s150/Baseline/" --dataset="MIBI2CH" --model_name="BS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_segCE_w0500_seed42" --saveout=True --scribbles=150 --gpu=1 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=10 --seed=42 --train=True  --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --nsaves=6 --ratio=0.95  --wfore=0.5 --wback=0.5 --epochs=400 --batch=64 --load=False > MIBI2CH_BS_2tasks_verbose.txt
```

- Denoiseg Model: file is /Denoiseg/main_denoiseg.py -> Here change the directory of the dataset files 

```
python main_denoiseg.py --basedir="/data/natalia/models/MIBI2CH/s150/DenoiSeg/" --dataset="MIBI2CH" --model_name="DS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_segCErecL2_w04501_seed42" --saveout=True --scribbles=150  --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=10 --seed=42 --train=True  --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="L2" --nsaves=6  --wfore=0.45 --wback=0.45 --wrec=0.1 --epochs=400 --batch=64 --load=False > MIBI2CH_DS_2tasks_verbose.txt
```
