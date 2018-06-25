# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:44:48 2018
Use to configure model path
@author: zkx-96
"""

class ConfigPath:pass

ConfigPath.local_data_path = "/media/minivision/OliverSSD/LiveBody"
ConfigPath.local_caffe_path = "/home/minivision/SoftWare/caffe-server"
ConfigPath.model_root_path = "/media/minivision/OliverSSD/LiveBody/select_best_result"
ConfigPath.deploy_root_path = "{}/deploy_lib".format(ConfigPath.local_caffe_path)
ConfigPath.out_root_path = ""

ConfigPath.best_model_path = ""

ConfigPath.best_verification_model_path = ""


ConfigPath.test_data_set = {

         'live-haar':{
           'imgs_folder' : "{}/Testset/Test7_xch/patches-haar/".format(ConfigPath.local_data_path),
           #save patch into bmp format
           'image_list'  : "{}/Testset/Test7_xch/Landmark_label_bmp.txt".format(ConfigPath.local_data_path)
           },
}
