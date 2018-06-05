# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:02:53 2018

@author: minivision
"""

from __future__ import print_function
import sys
sys.path.append('./python/caffe')
sys.path.append('./python')

import time
import os
import stat
import subprocess
import shutil


import caffe_pb_feature
import utility
import roc_curve_maker
from config_path import *

class gethorModelTest():
    def __init__(self,  select_date, test_set , test_batch_num=-1):
        self.test_set = test_set
        self.select_date = select_date
        self.model_path = "{}/{}".format(ConfigPath.model_root_path , self.select_date)
        self.test_batch_num = test_batch_num
        
    def testRun(self):
            model_info_pair  = self.selectModelDeployList()
            if len(model_info_pair) == 0:
                print("no caffemodel to test..")
                return 
            else:
                #set path
                out_path = '{}/Result_{}'.format(self.model_path,self.test_set)
                script_path = '{}/scripts/face_feature_scripts'.format(ConfigPath.local_caffe_path)
                utility.make_dirs(out_path)
                utility.make_dirs(script_path)
                
                #create param file
                for index, batch_model_info in enumerate(model_info_pair):
                    param = self.createParamFile(batch_model_info, out_path)
                    #save into files .prototxt
                    param_path = '{}/test_config_file_E{}.prototxt'.format(script_path, index)
                    f = open(param_path, 'w')
                    print(param, file = f)
                    f.close()    
                    
                    #create execute file .sh  --where test engine is
                    execute_path = '{}/verification_test_E{}.sh'.format(script_path, index)
                    f = open(execute_path, 'w')
                    f.write('cd {}\n'.format(ConfigPath.local_caffe_path))
                    f.write('./build/tools/face_feature_extractor.bin {}'.format(param_path))
                    f.close()

                    #execute file
                    os.chmod(execute_path, stat.S_IRWXU)
                    subprocess.call(execute_path, shell=True) 
                
                #copy scripts back to out path
            copy_path = '{}/face_feature_scripts'.format(self.model_path)
            if not os.path.exists(copy_path):
                shutil.copytree(script_path,'{}/face_feature_scripts'.format(self.model_path) )   
                 
            self.curvePrecious(out_path)
            
    def selectModelDeployList(self):
        folder_list = [elem for elem in os.listdir(self.model_path) if not elem.endswith('.txt') and not elem.endswith('.log') and not elem.endswith('.png') ]
        model_info_pair = {} #model name -- deploy file
        for folder in folder_list:
                model_list = []
                deploy_path = ""
                for root_path, folder_path, filename_path in os.walk('{}/{}'.format(self.model_path, folder)):
                    for filename in filename_path:
                        if filename.endswith('.caffemodel'):
                            print(filename)
                            model_list.append('{}/{}'.format(root_path, filename))
                        if filename.find('deploy.prototxt') >=0:
                            deploy_path = '{}/{}'.format(root_path, filename)
                            
                if deploy_path not in model_info_pair.keys() and len(deploy_path) != 0 and len(model_list) != 0 :   
                    model_info_pair[deploy_path] = model_list
        
        #divide model list info into batches..
        epoch_model_list = []
        model_info_item = model_info_pair.items()
        each_batch = []
        sum_num = len(model_info_item)
        
        if self.test_batch_num == -1:
             return [model_info_item]
        else:
            for i in range(1, sum_num+1):
               # print(model_info_item[i-1])
                each_batch.append(model_info_item[i-1])
                if i % self.test_batch_num == 0:
                    epoch_model_list.append(each_batch)
                    each_batch =  []
            epoch_model_list.append(each_batch) 
            
        return epoch_model_list
        
    def createParamFile(self, batch_model_info, output_result_path, mean_value = None, scale = 1.0):
            model_param_config = []
            for elem in batch_model_info:
                deploy_path = elem[0]
                model_lists = elem[1]
                for model_name_path in model_lists:
                    model_name = model_name_path.split('/')[-1]
                    patch_info = utility.crop_patch_info_model_name(model_name)
                    if mean_value != None:
                        each_param = caffe_pb_feature.ModelInitParameter(
                                    image_root_path = '{}/{}'.format(ConfigPath.test_data_set[self.test_set]['imgs_folder'], patch_info),
                                    deploy_path = deploy_path,
                                    model_path = model_name_path,
                                    output_path = output_result_path,
                                    data_transform = caffe_pb_feature.TransformationParameter(
                                        mean_value = mean_value,
                                        scale = scale
                                        )
                                    )
                    else:
                       each_param = caffe_pb_feature.ModelInitParameter(
                                    image_root_path = '{}/{}'.format(ConfigPath.test_data_set[self.test_set]['imgs_folder'], patch_info),
                                    deploy_path = deploy_path,
                                    model_path = model_name_path,
                                    output_path = output_result_path,
                                    )
                    model_param_config.append(each_param)
            
            feature = caffe_pb_feature.ExtractFeatureParameter(
            run_mode = caffe_pb_feature.ExtractFeatureParameter.RunMode.Value('GPU'),
                            device_id = 0,
                            image_list = ConfigPath.test_data_set[self.test_set]['image_list'],
                            image_pair_list = ConfigPath.test_data_set[self.test_set]['pairs_file'],
                            model_config = model_param_config
                        )           
            return feature
            
    #draw roc curve and statistic result
    def curvePrecious(self, output_result_path):
        statistic_info = []
        model_list = [elem for elem in os.listdir(output_result_path) if elem.find('result_v2') >=0 and elem.endswith('.txt')] #find selected date's model
        for filename in model_list:
           roc_image_name = filename.replace('.txt', '_ROC.png')
           #print(roc_image_name)
           if not os.path.exists('{}/{}'.format(output_result_path, roc_image_name)):
               print (filename)
               statistic_info.append(filename)
               roc_info = roc_curve_maker.roc_maker('{}/{}'.format(output_result_path, filename))
               statistic_info += roc_info
               statistic_info.append("\n===============================\n")

        date_minu = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        statistic_result_path = '{}/roc_statistic_result_{}.log'.format(output_result_path, date_minu)
        f = open(statistic_result_path, 'w')
        for line in statistic_info:
            f.write('{}\n'.format(line))
        f.close()
                    
        