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
import time



class DistillSoftmax(combineModelParam):
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
        
        
        #create add net name list
        each_net_last_layer_name = []
        for key in record_layer_index.keys():
            add_name = net_proto.layer[key-1].name
            each_net_last_layer_name.append(add_name)
            #delete last teacher softmax layer 
            if add_name.find("teacher") >=0:
               del net_proto.layer[key]             
        #add euclidean_loss_layer
        euclidean_proto = self.create_euclidean_loss_layer(each_net_last_layer_name)
        f = open(self.dst_model_path['dst_deploy'], 'w')
        print(net_proto, file = f)
        print(euclidean_proto, file = f)
        #print(softmax_proto, file = f)
        f.close()
        
    #each layer's name is a list
    def create_euclidean_loss_layer(self, each_net_last_layer_name):
            euclidean_loss_layer= caffe_pb2.LayerParameter(
            name = "euclidean_loss",
            type = "EuclideanLoss",
            bottom = each_net_last_layer_name
            )
            
            net_proto = caffe_pb2.NetParameter()
            net_proto.layer.extend([euclidean_loss_layer])
            return net_proto
            
     #add_net_layer_name is a list
    def create_softmax_layer(self, add_net_layer_name):
          add_softamx_layers = []       
          for elem in add_net_layer_name:   
                softmax_layer= caffe_pb2.LayerParameter(
                name = "{}-softmax".format(elem),
                type = "Softmax",
                bottom = elem
                )
                add_softamx_layers.append(softmax_layer)
                
          net_proto = caffe_pb2.NetParameter()
          for elem in add_softamx_layers:
              net_proto.layer.extend([elem])
          return net_proto
          
if __name__ == '__main__':
    
    date = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())) 
    root_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/HistoryBestModel"
    dst_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/Combined_model_distill/{}".format(date)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        
    patch_folder = ["2018-07-05_FakeFaceMtcnn_fc_0.4_80x80_fakeface3_DeepIDCReluC3_zkx_iter_55000",
    "2018-07-04_FakeFaceMtcnn_fc_0.4_80x80_fakeface3_MobileFaceNet-k-5-5_zkx_iter_20000" ]
    prefix_names = ["student", "teacher"]
    
    f = open('{}/net_info.txt'.format(dst_path), 'w')
    print(patch_folder, file = f)
    print(prefix_names, file = f)
    f.close()
    
    
    combine_model = DistillSoftmax(root_path,dst_path )
    combine_model.model_combination(patch_folder, prefix_names)