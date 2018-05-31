#use to change deploy's input
#Author Zkx@__@
#Date 2018-01-15

from __future__ import print_function
import sys
sys.path.append("./python")
from google.protobuf import text_format
import caffe_pb2
import os

def change_input(src_deploy_file, dst_deploy_file, height, width):

     print(height, width)
     net_proto = caffe_pb2.NetParameter()
     #load net
     f = open(src_deploy_file, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()

     net_proto.input_dim[2] = height
     net_proto.input_dim[3] = width

     #save changed model
     f = open(dst_deploy_file, 'w')
     print(net_proto, file=f)
     f.close()

     
     
#use to sort model_names by iteration number
def sort_model_name(model_name_list):
    dict_model_name = {}
    for line in  model_name_list:
           if line.find('iter') >=0 and line.find('.txt') >=0:
                model_name = line.split('.txt')[0]
                num = int(model_name.split('_')[-1])
                dict_model_name[num] = line
    key = sorted(dict_model_name.keys())
    print (key)
    sort_list = []
    for elem in key:
        #print (dict_model_name[elem])
        sort_list.append(dict_model_name[elem])
    return sort_list
    
    
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


#if __name__ == '__main__':
    #src_deploy_file = "./scripts/verification_script/deploy.prototxt"
    #dst_deploy_file = "./scripts/verification_script/chnaged_deploy.prototxt"
    #change_deploy_input(src_deploy_file, dst_deploy_file, 120,100)
