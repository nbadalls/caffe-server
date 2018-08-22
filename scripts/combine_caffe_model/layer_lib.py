
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
import caffe_pb2
from combine_model_param import *
import time
import os


#each layer's name is a list
def create_euclidean_loss_layer(each_net_last_layer_name, name_prefix = "",\
                                use_norm = True ):

        net_proto = caffe_pb2.NetParameter()
        euclidean_loss_bottom = each_net_last_layer_name
        if use_norm:
            euclidean_loss_bottom = []
            for each_net in each_net_last_layer_name:
                layer = caffe_pb2.LayerParameter(
                name = "l2" + each_net,
                type = "L2Norm",
                bottom = [each_net],
                top = ["l2" + each_net]
                )
                euclidean_loss_bottom.append("l2" + each_net)
                net_proto.layer.extend([layer])


        euclidean_loss_layer= caffe_pb2.LayerParameter(
        name = "euclidean_loss" + name_prefix,
        type = "EuclideanLoss",
        bottom = euclidean_loss_bottom,
        top = ["euclidean_loss" + name_prefix]
        )
        net_proto.layer.extend([euclidean_loss_layer])

        return net_proto

 #add_net_layer_name is a list
def create_softmax_layer(add_net_layer_name):
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
