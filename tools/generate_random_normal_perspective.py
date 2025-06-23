import os.path

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from euqiCubeTransform import e2c, e2p
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import random


fov = 60 # Change as needed
save_root_path = ''
save_folder_name = 'Stanford2D3D_FoV' + str(fov) + '_v30_rot0'

# read lines from txt
txt = '' # Replace with the actual path of your txt file # each row: image_path depth_path\n

save_txt_path = save_root_path + os.path.basename(txt).replace('Mac_', '').replace('.txt', 'fov'+str(fov)+'_v30_rot0_e2p.txt') # change file names as needed

with open(txt, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines] # delete '\n'


for line in lines:
    projection_list = []
    probs_list = []
    image_path = line.split(' ')[0]
    depth_path = line.split(' ')[1]
    file_name = os.path.basename(image_path)
    file_name_depth = os.path.basename(depth_path)
    file_dir_list = os.path.dirname(image_path).split('/')

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    for i in range(4):  # Change loop to generate 5 random projections (change as needed)
        u_deg = random.randint(-180, 180)
        v_deg = random.randint(-30, 30)
        rot_deg = 0

        projection = e2p.e2p(image, fov_deg=(fov, fov), u_deg=u_deg, v_deg=v_deg, out_hw=(256, 256), in_rot_deg=rot_deg,
                             mode='bilinear')

        save_path = save_root_path + file_dir_list[-2] + '/' + file_dir_list[-1] + '/' + file_name.replace('.png',
                                                                                                           '_' + str(
                                                                                                               u_deg) + '_' + str(
                                                                                                               v_deg) + '_' + str(
                                                                                                               rot_deg) + '.png')
        save_path = save_path.replace('/Stanford2D3D/', '/'+save_folder_name+'/')

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        cv2.imwrite(save_path, projection)

        projection_depth = e2p.e2p(depth, fov_deg=(fov, fov), u_deg=u_deg, v_deg=v_deg, out_hw=(256, 256), in_rot_deg=rot_deg,
                                   mode='bilinear')
        save_path_depth = save_root_path + file_dir_list[-2] + '/' + file_dir_list[-1] + '/' + file_name_depth.replace(
            '.exr',
            '_' + str(
                u_deg) + '_' + str(
                v_deg) + '_' + str(rot_deg) + '.exr')
        save_path_depth = save_path_depth.replace('/Stanford2D3D/', '/' + save_folder_name + '/')
        cv2.imwrite(save_path_depth, projection_depth)







