export PATH="/nadeem_lab/Gunjan/install/miniconda3/bin:$PATH"

# /nadeem_lab/Gunjan/install/miniconda3/envs/base_mlflow
# > /nadeem_lab/Gunjan/code/github-2023-09-04/ImPartial/experiments/impartial_tissuenet_all/log.txt


CUDA_VISIBLE_DEVICES=1 python train_impartial.py --experiment_name impartial-tissuenet_all --run_name tissuenet_all_ensemble_budget_0.2_scribble_modelsave --output_dir experiments/impartial_tissuenet_all/ensemble_budget_0.2_scribble_modelsav2/ --log_file_name log_train_ensemble.log --scribble_rate=0.2 --config config/tissuenet_all.ini 