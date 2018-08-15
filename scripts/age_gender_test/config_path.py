# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:44:48 2018
Use to configure model path
@author: zkx-96
"""

class ConfigPath:pass


ConfigPath.local_data_path = "/media/minivision/OliverSSD/GenderAge/Test/TestSet-patches"
ConfigPath.local_caffe_path = "/home/minivision/SoftWare/caffe-server"
ConfigPath.local_caffe_path2 = "/home/zkx/Project/O2N/caffe_reid_min"
ConfigPath.model_root_path = "/media/minivision/OliverSSD/GenderAge/best_models"
ConfigPath.out_root_path = "/home/zkx/Project/models/AgeGender/age_gender_result"
ConfigPath.best_model_path = "/home/zkx/Project/models/AgeGender/age_gender_best_select_model"
ConfigPath.best_verification_model_path = "/home/zkx/Project/models/AgeGender/age_gender_best_combine_model"



ConfigPath.test_data_set = {

         'age-gender1':{
           'imgs_folder' : "{}".format(ConfigPath.local_data_path),
           #save patch into bmp format
           'image_list'  : "{}/image_list/combine_list/combine_testset_Test1-4_2017_2018-08-09_softmax_label.txt".format(ConfigPath.local_data_path)
           },
        'age-gender1-b':{
          'imgs_folder' : "{}".format(ConfigPath.local_data_path),
          #save patch into bmp format
          'image_list'  : "{}/image_list/combine_list/combine_Test1-4+2017+2018-08-09_testset_softmax_label.txt".format(ConfigPath.local_data_path)
          },
     'age-gender1-b2':{
       'imgs_folder' : "{}".format(ConfigPath.local_data_path),
       #save patch into bmp format
       'image_list'  : "{}/image_list/combine_list/combine_2017+2018-08-09_testset_softmax_label.txt".format(ConfigPath.local_data_path)
       },

        'age-gender1-b3':{
          'imgs_folder' : "{}".format(ConfigPath.local_data_path),
          #save patch into bmp format
          'image_list'  : "{}/image_list/combine_list/2018-08-09_business_distract_testset_rename_GE0_list_softmax_label.txt".format(ConfigPath.local_data_path)
          },
}
