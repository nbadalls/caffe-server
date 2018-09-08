# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:12:01 2018

@author: minivision
"""
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
import os
from google.protobuf import text_format
import caffe_pb2
import layer_lib

#all deploy with same input size
#del_last_layer_num indice  delete num of last layers
#record_layer_index record each net's last layer's indice and name
#example: record_layer_index[1] = "softmax-1" record_layer_index[25] = "softmax-2"
def combine_single_deploy(nets_info, del_last_layer_num):
    net_proto = caffe_pb2.NetParameter()
    record_layer_index = {}
    num_nets = len(nets_info)
    #init  net proto
    if num_nets > 0:
        f = open(nets_info[0]['dstNet'], 'r')
        text_format.Merge(f.read(), net_proto)
        f.close()

        net_proto.input[0] = "data"
        for i in range(del_last_layer_num):
            del net_proto.layer[-1]
        #sum last number togther
        sum_layer_num = len(net_proto.layer)-1
        record_layer_index[sum_layer_num] = net_proto.layer[len(net_proto.layer)-1].name
        #concat first layer with "data"
        net_proto.layer[0].bottom[0] = "data"

      #write into each  layer
        for index in range(1, num_nets):
            net_proto_single = caffe_pb2.NetParameter()
            f = open(nets_info[index]['dstNet'], 'r')
            text_format.Merge(f.read(), net_proto_single)
            f.close()
            #concat all combine net name togther
            net_proto.name += "||{}".format(net_proto_single.name)
            #concat first layer with "data"
            net_proto_single.layer[0].bottom[0] = "data"

            for i in range(del_last_layer_num):
                del net_proto_single.layer[-1]

            sum_layer_num+=len(net_proto_single.layer)
            record_layer_index[sum_layer_num] = net_proto_single.layer[len(net_proto_single.layer)-1].name
            #combine into net proto
            for elem_layer in net_proto_single.layer:
                #net_proto.layer+= elem_layer
                 net_proto.layer.extend([elem_layer])
    return net_proto, record_layer_index


#use to  combine different models loaded by opencv
def combine_single_deploy_model_merge(nets_info, del_last_layer_num):
    net_proto = caffe_pb2.NetParameter()
    record_layer_index = {}
    num_nets = len(nets_info)
    sum_layer_num = 0
    #init  net proto
    if num_nets > 0:

        for index in range(0, num_nets):
            net_proto_single = caffe_pb2.NetParameter()
            f = open(nets_info[index]['dstNet'], 'r')
            text_format.Merge(f.read(), net_proto_single)
            f.close()

            #delete last n layer
            for i in range(del_last_layer_num):
                del net_proto_single.layer[-1]

            sum_layer_num+=len(net_proto_single.layer)
            record_layer_index[sum_layer_num] = net_proto_single.layer[len(net_proto_single.layer)-1].name
            #combine into net proto
            for elem_layer in net_proto_single.layer:
                #net_proto.layer+= elem_layer
                 net_proto.layer.extend([elem_layer])
    return net_proto, record_layer_index

def create_single_prefix_deploy(nets):
        for elem_net in nets:
            inputNet = elem_net['originalNet']
            addedPrefix = elem_net['prefix']
            outputNet = elem_net['dstNet']
            outputMap =  elem_net['outputLayerMap']

            with open(inputNet,'r') as f:
                originalNetSpec = f.read().splitlines()

            nameMap=[]
            for idx in xrange(len(originalNetSpec)):
                if ('name:' in originalNetSpec[idx]) or ('top:' in originalNetSpec[idx]) or ('bottom:' in originalNetSpec[idx]) or ('input:' in originalNetSpec[idx]):
                    originalText = originalNetSpec[idx].split(":")[-1].lstrip(" ")
                    newText = '"' + addedPrefix + originalText.lstrip('"')
                    originalNetSpec[idx] = originalNetSpec[idx].replace(originalText,newText)
                    print (originalNetSpec[idx])
                    if (('name:' in originalNetSpec[idx]) or ('input:' in originalNetSpec[idx])) and ('#' not in originalNetSpec[idx]):
                        nameMap.append(originalText.split('"')[1] + ',' + newText.split('"')[1])

            with open(outputNet,'w') as f:
                for line in originalNetSpec:
                    f.write("{}\n".format(line))

            with open(outputMap,'w') as f:
                for line in nameMap:
                    f.write("{}\n".format(line))
            f.close()


#Input model path information
def create_single_net(root_path, dst_path, patch_folder, prefix_name):
    #root_path = "/home/minivision/Work_File/Combine_Model/FakeFace/Combine"
    #patch_folder = ['FakeFace_fc_0.4_96x96_DeepID_S', 'FakeFace_le_0.3_80x80_DeepID', 'FakeFace_le_re_n_0.8_60x60_DeepID']
    #prefix_name = ['fc_0.4_96X96', 'le_0.3_80X80', 'le_re_n_0.8_60X60']
    model_path = []
    deploy_file_path = []
    for folder in patch_folder:
        abs_path = '{}/{}'.format(root_path, folder)
        for patch_root_path, folders, filenames in os.walk(abs_path):
            for filename in filenames:
                if filename.endswith(".caffemodel"):
                    model_path.append('{}/{}'.format(patch_root_path, filename))
                if filename == 'deploy.prototxt':
                    deploy_file_path.append('{}/{}'.format(patch_root_path, filename))

    nets = []
    dst_model_path = {}

    for i in range(len(patch_folder)):
        net_info = {}

        prefix_folder = '{}/{}/prefixed'.format(root_path, patch_folder[i])
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)
        #for rename deploy
        # net_info['originalNet'] = '{}/{}/deploy.prototxt'.format(root_path, patch_folder[i])
        net_info['originalNet'] = deploy_file_path[i]
        net_info['dstNet'] = '{}/{}/prefixed/prefixed_deploy.prototxt'.format(root_path, patch_folder[i])
        # net_info['prefix'] = '{}/'.format(prefix_name[i].replace('.', '_'))
        net_info['prefix'] = '{}/'.format(prefix_name[i])
        net_info['outputLayerMap'] = '{}/{}/prefixed/{}_layer_map.txt'.format(root_path, patch_folder[i], prefix_name[i])

        #for convert model
        net_info['net_configuration'] = net_info['originalNet']
        # net_info['pretrained_model'] = '{}/{}/{}'.format(root_path, patch_folder[i], model_name[i])
        net_info['pretrained_model'] = model_path[i]
        net_info['layer_map'] = net_info['outputLayerMap']
        nets.append(net_info)

    dst_model_path['dst_deploy'] = '{}/combine_{}_models_deploy.prototxt'.format(dst_path, len(nets))
    dst_model_path['dst_model'] = '{}/combine_{}_models.caffemodel'.format(dst_path, len(nets))
    return nets, dst_model_path
