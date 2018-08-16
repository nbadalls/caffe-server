# -*- coding: utf-8 -*-
"""
class face verification test
1.save face verification result into ConfigPath.out_root_path
2. Draw roc curve and statistic accuracy into log file              runTest[1-2]
"""
from __future__ import print_function
import sys
sys.path.append('./python/caffe')
sys.path.append('./python')

import time
import os
import stat
import subprocess
import caffe_pb2

import getPatchInfoFunc
import utility
import roc_curve_maker
from config_path import *


class ModelTest():

    def __init__(self, select_date, test_set_type, gpu_id, test_batch_num = -1):

        self.select_date = select_date
        self.test_set_type = test_set_type
        self.gpu_id = gpu_id
        self.test_batch_num = test_batch_num
        self.current_date = time.strftime('%Y-%m-%d', time.localtime())

    #model_path -- where caffemodel is
    #test_set_type -- select test set
    #test models includes in the model_path
    def runTest(self, model_path):

            self.model_path = model_path
            #find whether exists model to test  split models into batches
            select_model_list = self.selectModelList()

            if len(select_model_list[0]) == 0:
                #print("{} date's model is empty".format(self.select_date))
                return
            else:

                model_prefix = self.model_path.split('/')[-1]
                #get patch info
                begin = model_prefix.find('_')  #AMImageMeanCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet
                sign_pos = model_prefix.find('x')
                split_part2 = model_prefix[sign_pos:]#x96_b+FaceAdd_MobileFaceNet
                end = split_part2.find('_')
                patch_info = '{}{}'.format(model_prefix[begin+1 : sign_pos], split_part2[0:end])#fc_0.35_112 x96

                #set dst save path
                output_path = '{}/{}/{}/{}'.format(ConfigPath.out_root_path, self.current_date, self.test_set_type, model_prefix)
                output_deploy_path = '{}/train_file'.format(output_path)
                output_result_path = '{}/distance_result'.format(output_path)
                output_script_path = '{}/scripts'.format(output_path)
                utility.make_dirs(output_deploy_path)
                utility.make_dirs(output_result_path)
                utility.make_dirs(output_script_path)

                #train net
                net_name = model_prefix.split('_')[-1]
                deploy_file = '{}_deploy.prototxt'.format(net_name)
                src_deploy_path = '{}/{}'.format(ConfigPath.deploy_root_path, deploy_file)
                dst_deploy_path = '{}/{}'.format(output_deploy_path, deploy_file)

                # change deploy's input and save into dst path
                patch_info_dict = getPatchInfoFunc.splitPatchInfo(patch_info)
                utility.change_input(src_deploy_path, dst_deploy_path, patch_info_dict['height'], patch_info_dict['width'])

                #ceeate the config file
                print(model_prefix)
                for index, elem in enumerate(select_model_list):

                    #no means and scale
                    param = self.create_param( elem, patch_info, dst_deploy_path, output_result_path)
                    #save into files .prototxt
                    param_path = '{}/test_config_file_E{}.prototxt'.format(output_script_path, index)
                    f = open(param_path, 'w')
                    print(param, file = f)
                    f.close()

                    #create execute file .sh  --where test engine is
                    execute_path = '{}/verification_test_E{}.sh'.format(output_script_path, index)
                    f = open(execute_path, 'w')
                    f.write('cd {}\n'.format(ConfigPath.local_caffe_path))
                    f.write('./build/tools/face_feature_extractor.bin {}'.format(param_path))
                    f.close()

                    #execute file
                    os.chmod(execute_path, stat.S_IRWXU)
                    subprocess.call(execute_path, shell=True)

            #save roc curve result into log
            self.curvePrecious(output_result_path)


    #create test param according to model_list
    def create_param(self, model_list, patch_info, dst_deploy_path, output_result_path, mean_value = None, scale = 1.0):

        model_param_config = []
        for elem in model_list:

            if mean_value != None:
                each_param = caffe_pb2.ModelInitParameter(
                                    image_root_path = '{}/{}'.format(ConfigPath.test_data_set[self.test_set_type]['imgs_folder'], patch_info),
                                    deploy_path = dst_deploy_path,
                                    model_path = '{}/{}'.format(self.model_path, elem),
                                    output_path = output_result_path,
                                    data_transform = caffe_pb2.TransformationParameter(
                                        mean_value = mean_value,
                                        scale = scale
                                        )
                                    )
            elif elem.find("Means") >=0:
                each_param = caffe_pb2.ModelInitParameter(
                    image_root_path = '{}/{}'.format(ConfigPath.test_data_set[self.test_set_type]['imgs_folder'], patch_info),
                    deploy_path = dst_deploy_path,
                    model_path = '{}/{}'.format(self.model_path, elem),
                    output_path = output_result_path,
                    data_transform = caffe_pb2.TransformationParameter(
                     mean_value = [127.5, 127.5, 127.5],
                      scale = 128.0
                      )
                    )
            else:
                each_param = caffe_pb2.ModelInitParameter(
                                    image_root_path = '{}/{}'.format(ConfigPath.test_data_set[self.test_set_type]['imgs_folder'], patch_info),
                                    deploy_path = dst_deploy_path,
                                    model_path = '{}/{}'.format(self.model_path, elem),
                                    output_path = output_result_path,
                                    )
            model_param_config.append(each_param)

        feature = caffe_pb2.ExtractFeatureParameter(
        run_mode = caffe_pb2.ExtractFeatureParameter.RunMode.Value('GPU'),
                            device_id = self.gpu_id,
                            image_list = ConfigPath.test_data_set[self.test_set_type]['image_list'],
                            image_pair_list = ConfigPath.test_data_set[self.test_set_type]['pairs_file'],
                            model_config = model_param_config
                        )
        return feature


    #select which model to test
    def selectModelList(self):
        epoch_list = []
        model_list = [ elem for elem in os.listdir(self.model_path) if elem.find(".caffemodel") >=0 and elem.find(self.select_date) >=0]
        #test all models
        if self.test_batch_num == -1:
             epoch_list.append(model_list)
        else:

            sum_num =len(model_list)
            epoch_num = int(sum_num / self.test_batch_num)

            batch_list = []
            for i in range(1, epoch_num * self.test_batch_num+1):
                batch_list.append(model_list[i-1])
                if i % self.test_batch_num == 0:
                    epoch_list.append(batch_list)
                    batch_list = []
            for i in range(epoch_num * self.test_batch_num+1, sum_num+1):
                batch_list.append(model_list[i-1])
            epoch_list.append(batch_list)

        return epoch_list

    #draw roc curve and statistic result
    def curvePrecious(self, output_result_path):
        statistic_info = []
        model_list = [elem for elem in os.listdir(output_result_path) if elem.find('result_v2') >=0 and elem.endswith('.txt')] #find selected date's model
        sorted_model_list = utility.sort_model_name(model_list)
        for filename in sorted_model_list:
           roc_image_name = filename.replace('.txt', '_ROC.png')
           #print(roc_image_name)
           if not os.path.exists('{}/{}'.format(output_result_path, roc_image_name)):
               print (filename)
               statistic_info.append(filename)
               roc_info = roc_curve_maker.roc_maker('{}/{}'.format(output_result_path, filename))
               statistic_info += roc_info
               statistic_info.append("\n===============================\n")

        parent_path = os.path.abspath(os.path.join(output_result_path, ".."))
        date_minu = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        statistic_result_path = '{}/roc_statistic_result_{}.log'.format(parent_path, date_minu)
        f = open(statistic_result_path, 'w')
        for line in statistic_info:
            f.write('{}\n'.format(line))
        f.close()
