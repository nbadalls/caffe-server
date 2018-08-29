# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:52:53 2018

@author: minivision
"""
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
from caffe.proto import caffe_pb2
from combine_model_param import *
from layer_lib import *
import time
import os
import tkFileDialog
import shutil
import getPatchInfoFunc

class CVTransAgeGender(combineModelParam):
    def __init__(self, single_root_path, dst_combine_path):
        combineModelParam.__init__(self,single_root_path,dst_combine_path )

    def create_combine_deploy(self):

        data_net_proto = create_combine_data_layer(self.prefix_names)
        net_proto , record_layer_index= combine_utility.combine_single_deploy_model_merge(self.nets, 0)

        age_layer_name = []
        gender_layer_name = []
        patch_num = len(self.prefix_names)

        for key, value in record_layer_index.items():
            gender_layer_name.append(value)
            age_layer_name.append(net_proto.layer[key-3].name)
            print(value, net_proto.layer[key-3].name)


        if patch_num == 1:
            concat_proto = create_concat_layer([age_layer_name[0], gender_layer_name[0]])
            f = open(self.dst_model_path['dst_deploy'], 'w')
            print(data_net_proto, file = f)
            print(net_proto, file = f)
            print(concat_proto, file = f)
            f.close()

        else:

            age_eltwise_layer = create_eltwise_layer(age_layer_name, 'eltwise-age-softmax')
            age_scale = create_scale_layer( 'eltwise-age-softmax', 'eltwise-age-softmax-scale', 1.0/patch_num)
            gender_eltwise_layer = create_eltwise_layer(gender_layer_name, 'eltwise-gender-softmax')
            gender_scale = create_scale_layer( 'eltwise-gender-softmax', 'eltwise-gender-softmax-scale', 1.0/patch_num)
            concat_names = ['eltwise-age-softmax-scale', 'eltwise-gender-softmax-scale']
            concat_layer = create_concat_layer(concat_names)

            f = open(self.dst_model_path['dst_deploy'], 'w')
            print(data_net_proto, file = f)
            print(net_proto, file = f)
            print(age_eltwise_layer, file = f)
            print(age_scale, file = f)
            print(gender_eltwise_layer, file = f)
            print(gender_scale, file = f)
            print(concat_layer, file = f)
            f.close()


if __name__ == '__main__':

    root_path = "/media/minivision/OliverSSD/GenderAge/best_models/HistoryModel"
    parent_path = os.path.abspath(os.path.join(root_path, '..'))

    date = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    dst_combined_path = "{}/Combine_model/{}_AgeGender_business".format(parent_path, date)
    dst_combine_model_path = "{}/caffe_model".format(dst_combined_path)
    if not os.path.exists(dst_combine_model_path):
        os.makedirs(dst_combine_model_path)

    #get best combine list from file
    current_result_path = parent_path
    bestcombine_file_path = tkFileDialog.askopenfilename(initialdir =current_result_path, filetypes=[("text file", "*.log")])
    f = open(bestcombine_file_path, 'r')
    list_data = f.read().splitlines()
    f.close()

    #move best_combine_log into dst folder
    log_path = os.path.abspath(os.path.join(bestcombine_file_path, '..'))
    log_folder_name = log_path.split('/')[-1]
    shutil.copytree(log_path, dst_combined_path + '/' + log_folder_name)

    #patch_folder = ["2018-08-20_AgeGenderMtcnn_re_le_0.3_80x90_age-gender-dataset1_DeepID_zkx_iter_115000" ]
    patch_folder = [elem[elem.find(elem.split('_')[2]):].strip('.txt') for elem in list_data if elem.find('.txt') >=0]
    patch_info = [getPatchInfoFunc.crop_patch_info(elem) for elem in patch_folder]

    #add prefix to the patch info according the patch counter
    prefix_counter = {}
    prefix_names = []
    for elem in patch_info:
        if not elem in prefix_counter.keys():
            prefix_counter[elem] = 0
        else:
            prefix_counter[elem] += 1
        prefix_names.append('{}-{}'.format(elem, prefix_counter[elem]))


    f = open('{}/net_info.txt'.format(dst_combine_model_path), 'w')
    print(patch_folder, file = f)
    print(prefix_names, file = f)
    f.close()


    combine_model = CVTransAgeGender(root_path,dst_combine_model_path )
    combine_model.model_combination(patch_folder, prefix_names)
