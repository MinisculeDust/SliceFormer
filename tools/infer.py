import subprocess

if __name__ == '__main__':

    project_path = '' # project_path: absolute path to SliceFormer
    if project_path[-1] != '/':
        project_path += '/'

    config = {
        'trained_model': project_path + 'preTrainedModel/trained_sliceFormer.pth',
        'trained_model_high': project_path + 'preTrainedModel/trained_sliceFormer_high.pth',
        'data_path': project_path + '/data', # data_path: path to dataset
        'saved_data_path': project_path + 'outputs',
        'data_list': project_path + 'dataList/3D60_test_Stanford2D3D_area5a_5percent_.txt',
        'data_list_high': project_path + 'dataList/Pano3D_M3D_high_train_5percent.txt',
    }

    subprocess.run(['python3', project_path +'pyTorch/SliceFormer_Inference.py',
                    '--trained_model', config['trained_model_high'],
                    '--resolution', 'high',
                    '--data_path', config['data_path'],
                    '--saved_data_path', config['saved_data_path'],
                    '--filenames_file_eval', config['data_list_high']])
