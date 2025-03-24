# export PATH="/nadeem_lab/Gunjan/miniconda3/bin:$PATH"


# basedir_root="/nadeem_lab/Gunjan/experiments/Vectra_2CH/test_new_server/"


export PATH="/nadeem_lab/Gunjan/miniconda3/bin:$PATH"

  
basedir_root="/nadeem_lab/Gunjan/experiments/vectra_noEnsemble/"
log_root="/nadeem_lab/Gunjan/experiments/logs/vectra/20220131/noEnsemble_2task/"

# CUDA_VISIBLE_DEVICES=0 python3.9 main_impartial.py --basedir=$basedir_root"Vectra_2CH/s200/Impartial/" --dataset="Vectra_2CH" --model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42" --saveout=True --scribbles=200 --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=2 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=80 --batch=64 --load=False > /nadeem_lab/Gunjan/experiments/logs/Vectra_2CH/20211020/Vectra_2CH_Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42_verbose_s200.txt


CUDA_VISIBLE_DEVICES=2 python3.9 main_impartial.py --basedir=$basedir_root"Vectra_2CH/s200/Impartial/" --dataset="Vectra_2CH" --model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42" --saveout=True --scribbles=200 --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=1 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=300 --batch=64 --load=False > $log_root"Vectra_2CH_Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed42_verbose_s200.txt"






