import pandas as pd
import os
import sys
import shutil
sys.path.append("../")
from general.utils import mkdir,load_json,save_json


basedir_root = '/data/natalia/models/'
save_folder = 'to_send/' #folder that will contain the files to send
save_folder_images = 'to_send_outimages/' #folder that will contain the files to send

mkdir(basedir_root+save_folder)
mkdir(basedir_root+save_folder_images)

pd_model_folders = pd.read_csv('model_path.csv')

# print(pd_model_folders)
for ix in range(len(pd_model_folders)):

    model_dir = pd_model_folders.iloc[ix]['model_path']
    print('###### Model :', basedir_root + model_dir)
    if os.path.exists(basedir_root+model_dir):

        #create model folder in save folder path

        folder_destination = basedir_root+save_folder+model_dir
        print('output files destination ',folder_destination)
        mkdir(folder_destination)
        files = ['history.json', 'config.json','pd_summary_results.csv']
        for f in files:
            if os.path.exists(basedir_root+model_dir + f):
                shutil.copy(basedir_root+model_dir + f,folder_destination)
                print('-copying file: ',f)
            else:
                print('!! Warning : file ' , f, 'not found')
        print()
        ## Save history, config jsons and summary results
        # history = load_json(basedir_root+model_dir + 'history.json')
        # config_json = load_json(basedir_root+model_dir + 'config.json')
        # pd_summary = pd.read_csv(basedir_root+model_dir + 'pd_summary_results.csv')

        # pd_summary.to_csv(basedir_root+save_folder+model_dir+'pd_summary_results.csv',index = 0)
        # save_json(history,basedir_root+save_folder+model_dir + 'history.json')
        # save_json(config_json, basedir_root + save_folder + model_dir + 'config.json')

        #save output images
        folder_destination_images = basedir_root + save_folder_images + model_dir
        mkdir(folder_destination_images)

        if os.path.exists(basedir_root+model_dir+'output_images/'):
            mkdir(folder_destination_images+'output_images/')
            print('output images destination : ', folder_destination_images + 'output_images/')
            for f in os.listdir(basedir_root+model_dir+'output_images/'):
                shutil.copy(basedir_root+model_dir+'output_images/' + f, folder_destination_images+'output_images/')
                print('-copying file: ','output_images/' + f)

        print()
        print('-------------------------')


    else:
        print(model_dir, ' not found ')
        print()
        print('-------------------------')