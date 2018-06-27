# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:44:48 2018
Use to configure model path
@author: zkx-96
"""

class ConfigPath:pass

ConfigPath.local_data_path = "/home/zkx/Data_sdb" 
ConfigPath.local_caffe_path = "/home/zkx/Project/O2N/caffe-master"
ConfigPath.model_root_path = "/home/zkx/Project/O2N/asset/snapshot"
ConfigPath.deploy_root_path = "{}/deploy_lib".format(ConfigPath.local_caffe_path)
ConfigPath.out_root_path = "/home/zkx/Project/O2N/verification_result_caffe-master"

ConfigPath.best_model_path = "/home/zkx/Project/O2N/best_select_models_caffe-master"

ConfigPath.best_verification_model_path = "/home/zkx/Project/O2N/verification_select_best_models_caffe-master"



ConfigPath.test_data_set = {

         'XCH':{
           'pairs_file' : "{}/TestSet/XCH_PAD_01_23/image_pair_list.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/TestSet/XCH_PAD_01_23/patches/XCH/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/TestSet/XCH_PAD_01_23/image_list.txt".format(ConfigPath.local_data_path)
           },

	   'XCH-mtcnn':{
           'pairs_file' : "{}/TestSet/O2N/XCH-small/image_list-mtcnnpair.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/TestSet/O2N/XCH-small/patches_mtcnn/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/TestSet/O2N/XCH-small/image_list-mtcnn.txt".format(ConfigPath.local_data_path)
           },

        'THL':
        {
            'image_list' : "{}/TestSet/Small_THL_ID_LP_Filtered_09_26/image_list.txt".format(ConfigPath.local_data_path),
            'pairs_file' : "{}/TestSet/Small_THL_ID_LP_Filtered_09_26/small_image_pairs_09_26.txt".format(ConfigPath.local_data_path),
            'imgs_folder' : "{}/TestSet/Small_THL_ID_LP_Filtered_09_26/patches".format(ConfigPath.local_data_path)
        },
         'XCH-mtcnn-b':{
           'pairs_file' : "{}/TestSet/XCH_PAD_01_23/XCH-big-mtcnn_listpair.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/TestSet/XCH_PAD_01_23/patches-mtcnn/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/TestSet/XCH_PAD_01_23/XCH-big-mtcnn_list.txt".format(ConfigPath.local_data_path)
           },
}
