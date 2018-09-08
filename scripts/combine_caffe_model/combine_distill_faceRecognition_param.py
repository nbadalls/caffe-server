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


class DistillfaceRecognition(combineModelParam):
    def __init__(self, single_root_path, dst_combine_path):
        combineModelParam.__init__(self,single_root_path,dst_combine_path )

    def create_combine_deploy(self):

        net_proto , record_layer_index= combine_utility.combine_single_deploy(self.nets, 1)

        #adjust teacher net's learning rate to 0
        for  elem_layer in net_proto.layer:
            if elem_layer.name.find("teacher") >=0:
                for elem_param in  elem_layer.param:
                        elem_param.lr_mult = 0
                        elem_param.decay_mult = 0


        f = open(self.dst_model_path['dst_deploy'], 'w')
        print(net_proto, file = f)
        f.close()


if __name__ == '__main__':

    date = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    # root_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/HistoryBestModel"
    root_path = "/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/combine_disdill/2018-08-28-1"
    dst_path = "/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/combine_disdill/2018-08-28-1/combine_model"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    patch_folder = ["2018-05-07_AdditMarginCdata-b0.35s30_fc_0.35_112x96_b+asian+cap10+pos+beid-MS_faceNet-20-light2s4-bn_zkx_iter_190000",
    "2018-08-03_AMImageMtcnn-b0.3s30_fc_0.35_112x96_clean-b+add1+2-1-delAsia-b3-P0.0_MobileFaceNet-bn_zkx_iter_165000" ]
    prefix_names = ["student", "teacher"]

    f = open('{}/net_info.txt'.format(dst_path), 'w')
    print(patch_folder, file = f)
    print(prefix_names, file = f)
    f.close()


    combine_model = DistillfaceRecognition(root_path,dst_path )
    combine_model.model_combination(patch_folder, prefix_names)
