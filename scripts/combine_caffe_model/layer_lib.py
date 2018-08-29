
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
from caffe.proto import caffe_pb2
from combine_model_param import *
import getPatchInfoFunc
import time
import os


def create_combine_data_layer(prefix_name): #prefix_name is patch list examples: [fc_0.4_60x60 ,le_0.3_50x50, re_0.1_40x50]
    #create affine param
    affine_info = caffe_pb2.AffineCropParameter(
        show_output_image = False
    )
    for patch_info in prefix_name:
        #use to divide same patch models, add fc_0.35_100x100-1, fc_0.35_100x100-2
        #exp: fc_0.35_100x100_iter_1000 & fc_0.35_100x100_iter_2000
        if patch_info.find('-') > 0:
            patch_info = patch_info.split('-')[0]

        patch_info_dict = getPatchInfoFunc.splitPatchInfo(patch_info)

        image_info_ = caffe_pb2.ImageInfo(
            height = patch_info_dict['height'],
            width = patch_info_dict['width']
        )

        single_affine = caffe_pb2.AffineImageParameter(
            center_ind = patch_info_dict['center_id'],
            center_ind_num = len(patch_info_dict['center_id']),
            norm_mode = caffe_pb2.AffineImage_Norm_Mode.Value('RECT_LE_RE_LM_RM'),
            norm_ratio = patch_info_dict['norm_ratio'],
            fill_type = False,
            value = 0,
            image_info = image_info_
        )
        affine_info.affine_image_param.extend([single_affine])

    #create data layer
    data_prefix_name = []
    for name in prefix_name:
        # data_prefix_name.append('{}/data'.format(name.replace('.', '_')))
        data_prefix_name.append('{}/data'.format(name))

    data_layer= caffe_pb2.LayerParameter(
        name = "affine_Crop",
        type = "AffineCrop",
        bottom = ["data", "landmark_position"],
        top = data_prefix_name,
        affine_crop_param = affine_info
    )

    #create net
    net = caffe_pb2.NetParameter(
        input = ['data', 'landmark_position'],
        input_dim = [1,3,60,60, 1,1,2,5]
    )
    net.layer.extend([data_layer])
    return net


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

def create_concat_layer(each_net_last_layer_name):
    concat_layer= caffe_pb2.LayerParameter(
        name = "l2",
        type = "Concat",
        bottom = each_net_last_layer_name,
        top = ['l2'],
    )
    net = caffe_pb2.NetParameter()
    net.layer.extend([concat_layer])
    return net

def create_eltwise_layer(each_net_last_layer_name, out_name):
    eltwise_layer= caffe_pb2.LayerParameter(
        name = out_name,
        type = "Eltwise",
        bottom = each_net_last_layer_name,
        top = [out_name],
        eltwise_param = dict(operation = caffe_pb2.EltwiseParameter.EltwiseOp.Value('SUM'))
    )
    net = caffe_pb2.NetParameter()
    net.layer.extend([eltwise_layer])
    return net

def create_scale_layer(input, out_name, scale):
    concat_layer= caffe_pb2.LayerParameter(
        name = out_name,
        type = "Scale",
        bottom = [input],
        top = [out_name],
        scale_param = caffe_pb2.ScaleParameter(
            filler = caffe_pb2.FillerParameter(
            type = "constant",
            value = scale
            )
        )
    )
    net = caffe_pb2.NetParameter()
    net.layer.extend([concat_layer])
    return net
