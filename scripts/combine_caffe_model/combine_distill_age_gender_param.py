# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:52:53 2018

@author: minivision
"""
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
import caffe_pb2
from combine_model_param import *
from layer_lib import *
import time
import os
from getPatchInfoFunc import *


class DistillAgeGender(combineModelParam):
    def __init__(self, single_root_path, dst_combine_path):
        combineModelParam.__init__(self,single_root_path,dst_combine_path )

    def create_combine_deploy(self):

        net_proto , record_layer_index= combine_utility.combine_single_deploy(self.nets, 0)

        #adjust teacher net's learning rate to 0
        for  elem_layer in net_proto.layer:
            if elem_layer.name.find("teacher") >=0:
                for elem_param in  elem_layer.param:
                        elem_param.lr_mult = 0
                        elem_param.decay_mult = 0


        #create added euclidean_loss_layer bottom list
        gender_fc_layer_name = []
        age_fc_layer_name = []
        for key in record_layer_index.keys():
            gen_fc = net_proto.layer[key-1].name
            age_fc = net_proto.layer[key-3].name
            gender_fc_layer_name.append(gen_fc)
            age_fc_layer_name.append(age_fc)

        #del softmax layer
        for key in record_layer_index.keys():
            if net_proto.layer[key].name.find('teacher') >=0:
               del net_proto.layer[key]
               del net_proto.layer[key-2]

        #add euclidean_loss_layer
        euclidean_proto_age = create_euclidean_loss_layer(age_fc_layer_name, "_age")
        euclidean_proto_gender = create_euclidean_loss_layer(gender_fc_layer_name, "_gender")
        f = open(self.dst_model_path['dst_deploy'], 'w')
        print(net_proto, file = f)
        print(euclidean_proto_age, file = f)
        print(euclidean_proto_gender, file = f)
        #print(softmax_proto, file = f)
        f.close()


if __name__ == '__main__':

    date = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    # root_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/HistoryBestModel"
    root_path = "/media/minivision/OliverSSD/GenderAge/best_models/distill"
    dst_path = "/media/minivision/OliverSSD/GenderAge/best_models/Combined_model_distill/{}".format(date)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    patch_folder = ["2018-08-14_AgeGenderMtcnn_re_0.2_80x50_age-gender-dataset1_DeepID_zkx_iter_125000",
    "2018-08-14_AgeGenderMtcnn_re_0.2_80x50_age-gender-dataset1_MobileFaceNet-k-5-4_zkx_iter_105000" ]
    prefix_names = ["student", "teacher"]

    f = open('{}/net_info.txt'.format(dst_path), 'w')
    print(patch_folder, file = f)
    print(prefix_names, file = f)
    f.close()


    combine_model = DistillAgeGender(root_path,dst_path )
    combine_model.model_combination(patch_folder, prefix_names)
