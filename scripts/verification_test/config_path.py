# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:44:48 2018
Use to configure model path
@author: zkx-96
"""

class ConfigPath:pass

ConfigPath.local_data_path = "/mnt/glusterfs/o2n/FaceRecognition" 
ConfigPath.local_caffe_path = "/home/hjg/Project/O2N/caffe-master"
ConfigPath.model_root_path = "/home/hjg/Project/O2N/asset/snapshot"
ConfigPath.deploy_root_path = "{}/deploy_lib".format(ConfigPath.local_caffe_path)
ConfigPath.out_root_path = "/home/hjg/Project/O2N/verification_result_caffe-master"

ConfigPath.best_model_path = "/home/hjg/Project/O2N/best_select_models_caffe-master"

ConfigPath.best_verification_model_path = "/home/hjg/Project/O2N/verification_select_best_models_caffe-master"


ConfigPath.test_data_set = {

         'XCH':{
           'pairs_file' : "{}/Test_Data/O2N/XCH-small/image_pair_list.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/Test_Data/O2N/XCH-small/patches/XCH/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/Test_Data/O2N/XCH-small/image_list.txt".format(ConfigPath.local_data_path)
           },
           
         'XCH-Ad':{
           'pairs_file' : "{}/Test_Data/O2N/XCH-small/image_pair_list.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/Test_Data/O2N/XCH-small/patches/XCH-Ad/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/Test_Data/O2N/XCH-small/image_list.txt".format(ConfigPath.local_data_path)
           },

	   'XCH-mtcnn':{
           'pairs_file' : "{}/Test_Data/O2N/XCH-small/image_list-mtcnnpair.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/Test_Data/O2N/XCH-small/patches_mtcnn/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/Test_Data/O2N/XCH-small/image_list-mtcnn.txt".format(ConfigPath.local_data_path)
           },

               'XCH-mtcnn-out':{
           'pairs_file' : "{}/Test_Data/O2N/XCH-small-outdoor/XCH-small-outdoor_listpair.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/Test_Data/O2N/XCH-small-outdoor/patches_mtcnn/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/Test_Data/O2N/XCH-small-outdoor/XCH-small-outdoor_list.txt".format(ConfigPath.local_data_path)
           },

                          'XCH-mtcnn-out2':{
           'pairs_file' : "{}/Test_Data/O2N/XCH-small-outdoor2/XCH-small-mtcnn_listpair.txt".format(ConfigPath.local_data_path),
           'imgs_folder' : "{}/Test_Data/O2N/XCH-small-outdoor2/patches_mtcnn/".format(ConfigPath.local_data_path),
           'image_list'  : "{}/Test_Data/O2N/XCH-small-outdoor2/XCH-small-mtcnn_list.txt".format(ConfigPath.local_data_path)
           },

           
}
