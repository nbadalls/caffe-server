from __future__ import print_function
import sys
sys.path.append('./python')
sys.path.append('./python/caffe')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from change_mobilefaceNet_SeNet import *

def add_SE_part_into_faceNet2s4m(net_deploy_path, dst_deploy_path):

     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()
     f = open(net_deploy_path, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()


     is_add_net = False
     add_net_top = ""
     change_res_name = ""
     SeNet_proto = caffe_pb2.NetParameter()
     #copy original net into SeNet_proto
     for index, elem_layer in enumerate(net_proto.layer):
                    next_index = index +1
                    if next_index > len(net_proto.layer)-1:
                    	next_index = len(net_proto.layer)-1
                    #insert SE uint
                    # print(index, elem_layer.name, net_proto.layer[next_index].type)            
                    if elem_layer.name.find("relu") >=0 and net_proto.layer[next_index].type == "Eltwise":

                         SeNet_proto.layer.extend([elem_layer])

                         #get layer param
                         #conv2_1_em/scale
                         name = elem_layer.name
                         part_name = name.split("relu")[1]
                         conv_num = int(part_name.split('_')[0])
                         conv_index = int(part_name.split('_')[1])
                         # print(conv_num, conv_index)


                         #get channel number from convolution layer (index-2)
                         channels = int(net_proto.layer[index-1].convolution_param.num_output)
                         cat_layer_name1 = SeNet_proto.layer[-1].top[0]
                         if conv_num == 1:
                         	cat_layer_name2 = "conv1_1"
                         elif conv_index == 1:
                         	cat_layer_name2 = "pool{}".format(conv_num-1)
                         else:
                         	cat_layer_name2 = add_net_top

                         #create SE branch
                         add_net = caffe.NetSpec()
                         add_net[elem_layer.top[0]],  add_net[cat_layer_name1], add_net[cat_layer_name2]= L.ImageDataParameter(ntop = 3)
                         Axpy_name = create_SE_part(add_net, elem_layer.top[0], conv_num,conv_index,channels, cat_layer_name1, cat_layer_name2)
                         add_net_proto = add_net.to_proto()
                         #delete data layer
                         del add_net_proto.layer[0]

                         #copy add branch into SE proto
                         for add_layer in add_net_proto.layer:
                               SeNet_proto.layer.extend([add_layer])

                         is_add_net = True
                         #save the top of SE branch (Axpy [scale + elwise] included )
                         add_net_top = Axpy_name
                    else:
                                   #Do not copy eltwise layer
                                   if elem_layer.type == "Eltwise":
                                         continue
                                   else:
                                        SeNet_proto.layer.extend([elem_layer])

                                   #change other net names
                                   if SeNet_proto.layer[-1].bottom[0] == change_res_name:
                                        print(SeNet_proto.layer[-1].name, change_res_name, add_net_top)
                                        del SeNet_proto.layer[-1].bottom[0]
                                        SeNet_proto.layer[-1].bottom.extend([add_net_top])


                                   if is_add_net:
                                   	#change bottom name
                                                  change_res_name = SeNet_proto.layer[-1].bottom[0]
                                                  del SeNet_proto.layer[-1].bottom[0]
                                                  SeNet_proto.layer[-1].bottom.extend([add_net_top])
                                                  is_add_net = False

     SeNet_proto.input.extend([net_proto.input[0]])
     SeNet_proto.input_dim .extend([net_proto.input_dim[0], net_proto.input_dim[1], net_proto.input_dim[2], net_proto.input_dim[3]])

     with open(dst_deploy_path, 'w') as f:
	print(SeNet_proto, file = f)



if __name__ == '__main__':
	net_deploy_path = "./deploy_lib/faceNet-20-light2s4m-bn_deploy.prototxt"
	dst_deploy_path = "./modify_deploy/SE-faceNet-20-light2s4m-bn_deploy.prototxt"
	add_SE_part_into_faceNet2s4m(net_deploy_path, dst_deploy_path)