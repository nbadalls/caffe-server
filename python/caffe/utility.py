#caffe net common utility
#Author zkx@__@
#Date 2018-01-31

from __future__ import print_function
import sys
sys.path.append('./python')
import os
import caffe_pb2
from google.protobuf import text_format

#find most date among the snapshot models
def find_most_recent_date(snapshot_folder):
    max_month = 0
    max_day = 0
    max_year = 0
    model_list = os.listdir(snapshot_folder)
    for elem in model_list:
        if elem.endswith(".caffemodel"):
            basename = os.path.basename(elem)
        else:
            continue
        #divide year month day
        date = basename.split('_')[0] #the first part of the model name is date
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])
        day = int(date.split('-')[2])

        if year > max_year:
            max_year = year
            max_month = month
            max_day = day
        elif year == max_year:
            if month > max_month:
                max_month = month
                max_day = day
            elif month == max_month:
                if day > max_day:
                    max_day = day
    return max_year, max_month, max_day


def find_biggest_iter_num(snapshot_folder, select_date):
    max_iter = 0
    model_list = os.listdir(snapshot_folder)
    for elem in model_list:
        if elem.endswith('.caffemodel'):
            basename = os.path.basename(elem).split('.caffemodel')[0]
            date = basename.split('_')[0]
            if select_date == date:
                iter_num = int(basename.split('_iter_')[-1])
                if iter_num > max_iter:
                    max_iter = iter_num

    return max_iter

def read_deploy_into_proto(deploy_file, net_proto_layer):
     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()

     f = open(deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

    #  #delete data layer
    #  del net_proto.layer[0]
     #delete slience layer
     del net_proto.layer[-1]
     #delete normalize layer
     del net_proto.layer[-1]

     #if last layer is batchnorm layer then set use_global_stats to false
     for elem in net_proto.layer: 
         if elem.type.find('BatchNorm') >=0 :
             elem.batch_norm_param.use_global_stats = False

     for j in range(len(net_proto.layer)):
         net_proto_layer.layer.extend([net_proto.layer[j]])
     return net_proto_layer

def read_deploy_into_proto_delete_slience(deploy_file, net_proto_layer):
     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()

     f = open(deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

    #  #delete data layer
    #  del net_proto.layer[0]
     #delete slience layer
     del net_proto.layer[-1]
     
     for j in range(len(net_proto.layer)):
         net_proto_layer.layer.extend([net_proto.layer[j]])
     return net_proto_layer

def read_deploy_into_proto_delete_slience_norm(deploy_file, net_proto_layer, bn_use_global_stats):

     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()

     f = open(deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

     #reset all batchnorm use_global_stats to false:
     for elem in net_proto.layer:
          if elem.type == "BatchNorm":
             elem.batch_norm_param.use_global_stats = bn_use_global_stats

    #  #delete data layer
    #  del net_proto.layer[0]
     #delete slience layer
     del net_proto.layer[-1]
     del net_proto.layer[-1]
     #if last layer is batchnorm layer then set use_global_stats to false
     #if net_proto.layer[-1].name.find('_bn') >=0 :
            #net_proto.layer[-1].batch_norm_param.use_global_stats = False

     for j in range(len(net_proto.layer)):
         net_proto_layer.layer.extend([net_proto.layer[j]])
     return net_proto_layer


def read_deploy_into_proto_changedp(deploy_file, net_proto_layer, bn_use_global_stats):

     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()

     f = open(deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

     for index, elem_layer in enumerate(net_proto.layer):
	if elem_layer.type == "Softmax":
		del net_proto.layer[index]

     #reset all batchnorm use_global_stats to false:
     for elem in net_proto.layer:
          if elem.type == "BatchNorm":
             elem.batch_norm_param.use_global_stats = bn_use_global_stats
          if elem.name.find("dw") >=0 and elem.type == "Convolution":
                elem.type = "DepthwiseConvolution"

     for j in range(len(net_proto.layer)):
         net_proto_layer.layer.extend([net_proto.layer[j]])
     return net_proto_layer

def read_deploy_into_proto_config_delete_num(deploy_file, net_proto_layer, bn_use_global_stats, delete_num):

     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()

     f = open(deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

     #reset all batchnorm use_global_stats to false:
     for elem in net_proto.layer:
          if elem.type == "BatchNorm":
             elem.batch_norm_param.use_global_stats = bn_use_global_stats

     for i in range(delete_num):
        del net_proto.layer[-1]

     for j in range(len(net_proto.layer)):
         net_proto_layer.layer.extend([net_proto.layer[j]])
     return net_proto_layer