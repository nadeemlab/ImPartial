export PATH="/nadeem_lab/Gunjan/miniconda3/bin:$PATH"

  
# basedir_root="/nadeem_lab/Gunjan/experiments/Cellpose_20211128_noEnsemble/"
# log_root="/nadeem_lab/Gunjan/experiments/logs/cellpose/20211128_noEnsemble/"

basedir_root="/nadeem_lab/Gunjan/experiments/cellpose_manual_scribble_secIter/"
# log_root="/nadeem_lab/Gunjan/experiments/logs/cellpose/20220118/noEnsemble_manual_scribble_noValidation/"
log_root="/nadeem_lab/Gunjan/experiments/logs/cellpose/20220124/noEnsemble_manual_scribble_noValidation_secIter/"



# CUDA_VISIBLE_DEVICES=1 python3.9 main_impartial.py --basedir=$basedir_root"cellpose/s200/Impartial/" --dataset="cellpose" --model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed42" --saveout=True --scribbles=200 --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=42 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=1 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=300 --batch=64 --load=False > $log_root"cellposeIm_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed42_verbose_s200.txt"




CUDA_VISIBLE_DEVICES=2 python3.9 main_impartial.py --basedir=$basedir_root"cellpose/manual_scribble/Impartial/" --dataset="cellpose_manual_scribble" --model_name="Im_1tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed43" --saveout=True --scribbles='manual_scribble' --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=43 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=1 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=300 --batch=64 --load=False > $log_root"cellpose_Im_1tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed43_verbose_noValidation_manual_scribble_secIter.txt"





# CUDA_VISIBLE_DEVICES=2 python3.9 main_impartial.py --basedir=$basedir_root"cellpose/s200/Impartial/" --dataset="cellpose" --model_name="Im_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed44" --saveout=False --scribbles=200 --gpu=0 --optim_regw=0.0001 --optim="adam" --lr=0.0005 --gradclip=0 --seed=44 --train=True --udepth="4" --ubase="64" --activation="relu" --batchnorm=False --seg_loss="CE" --rec_loss="gaussian" --nsaves=1 --mcdrop=True --reset_optim=True --reset_validation=False  --wfore=0.45 --wback=0.45 --wrec=0.1 --wreg=0.0 --ratio=0.95  --epochs=300 --batch=64 --load=False > $log_root"cellposeIm_2tasks_base64depth4relu_adam5e4_mcdrop1e4_nsave1_segCEGauss_w04501_seed44_verbose_s200.txt"




