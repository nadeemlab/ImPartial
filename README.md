

## Data Folders:

* /MIBI_2CH : 4  2-channel mibi images, segmentation classes: cytoplasm, nuclei in cytoplasm, nuclei out of cytoplasm
* /Vectra_2CH: 8  2-channel Vector images, segmentation classes: cytoplasm, nuclei in cytoplasm, nuclei out of cytoplasm

In each dataset folder make sure to change the ‘input_dir’ column values in the .csv files for the correct path of the folder.
(e.g., 
data_dir = ‘/Data/MIBI_2CH/‘
pd_file = pd.read_csv(data_dir+'files.csv',index_col=0)
pd_file['input_dir'] = data_dir
pd_file.to_csv(data_dir + 'files.csv')
)




## Examples for running main files :

#### Dataset MIBI_2CH with 150 scribbles:

- Impartial Model : file is /Impartial/main_impartial.py -> Here change the directory of the dataset files (line ~90)

python main_impartial.py --basedir="/data/natalia/models/MIBI2CH/s150/Impartial/" --dataset="MIBI2CH" --model_name="Im_2tasks_base64depth4relu_adam5e4_nsave6_segCEGauss_w04501_seed42" --scribbles=150 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=6 --reset_optim=True  --wfore=0.45 --wback=0.45 --wrec=0.1 --ratio=0.95  --epochs=400 --batch=64 --load=False > MIBI2CH_Im_2tasks_verbose.txt

- MumfordShah Model : file is /MumfordShah/main_ms.py -> Here change the directory of the dataset files 

python main_ms.py --basedir="/data/natalia/models/MIBI2CH/s150/MS/" --dataset="MIBI2CH" --model_name="MS_2tasks_base64depth4relu_adam5e4_nsave6_segCErecL2_w04501_seed42" --saveout=True --scribbles=150 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="L2" --nsaves=6 --reset_optim=True --wfore=0.45 --wback=0.45 --wrec=0.09 --wreg=0.01 --epochs=400 --batch=64 --load=False > MIBI2CH_MS_2tasks_verbose.txt

- Baseline Model : file is /Baseline/main_bs.py -> Here change the directory of the dataset files 

python main_bs.py --basedir="/data/natalia/models/MIBI2CH/s150/Baseline/" --dataset="MIBI2CH" --model_name="BS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_segCE_w0500_seed42" --saveout=True --scribbles=150 --gpu=1 --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=10 --seed=42 --train=True  --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --nsaves=6 --ratio=0.95  --wfore=0.5 --wback=0.5 --epochs=400 --batch=64 --load=False > MIBI2CH_BS_2tasks_verbose.txt

- Denoiseg Model: file is /Denoiseg/main_denoiseg.py -> Here change the directory of the dataset files 

python main_denoiseg.py --basedir="/data/natalia/models/MIBI2CH/s150/DenoiSeg/" --dataset="MIBI2CH" --model_name="DS_2tasks_base64depth4relu_adam5e4_gclip10_nsave6_segCErecL2_w04501_seed42" --saveout=True --scribbles=150  --optim_regw=0 --optim="adam" --lr=0.0005 --gradclip=10 --seed=42 --train=True  --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="L2" --nsaves=6  --wfore=0.45 --wback=0.45 --wrec=0.1 --epochs=400 --batch=64 --load=False > MIBI2CH_DS_2tasks_verbose.txt
