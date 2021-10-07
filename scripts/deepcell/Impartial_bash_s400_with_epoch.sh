export PATH="/cluster/home/shrivasg/miniconda3/bin:$PATH"


basedir_root="/nadeem_lab/Gunjan/experiments/deepcell/allmodels_with_epoch_fullrun/"

CUDA_VISIBLE_DEVICES=0 python3.8 main_impartial.py --basedir=$basedir_root"Deepcell/s400/Impartial/" --dataset="Deepcell" --model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42" --saveout=True --scribbles=400 --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=5 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=300 --batch=64 --load=False > /lab/deasylab1/Saad/Gunjan/code/Impartial-Pipeline/scripts/deepcell/logs/20200914/Deepcell_Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave5_segCEGauss_w04501_seed42_verbose_s400.txt

