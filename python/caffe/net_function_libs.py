import os
import sys
sys.path.append('./python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from caffe.proto import caffe_pb2


def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

#conv + prelu
def conv_prelu(net, from_layer,out_layer,lr_mult1, decay_mult1, lr_mult2, decay_mult2,
                num_output_, kernel_size_, weight_std, bias_value):
    kwargs = {
        'param': [dict(lr_mult = lr_mult1, decay_mult = decay_mult1),
         dict(lr_mult = lr_mult2, decay_mult = decay_mult2)],
        'convolution_param': dict(num_output = num_output_, kernel_size = kernel_size_,
          weight_filler = dict(type = "gaussian", std = weight_std),
          bias_filler = dict(type = "constant", value = bias_value))
    }

    conv_name = out_layer
    net[conv_name] = L.Convolution(net[from_layer], **kwargs)

    prelu_name = 'relu{}'.format(conv_name[-1])
    net[prelu_name] = L.PReLU(net[conv_name], prelu_param = dict(filler = dict(type = "gaussian", std=0.03 )), in_place=True )
    return prelu_name

#conv + crelu
def conv_crelu(net, from_layer,out_layer,lr_mult1, decay_mult1, lr_mult2, decay_mult2,
                num_output_, kernel_size_, weight_std, bias_value):
    kwargs = {
        'param': [dict(lr_mult = lr_mult1, decay_mult = decay_mult1),
         dict(lr_mult = lr_mult2, decay_mult = decay_mult2)],
        'convolution_param': dict(num_output = num_output_, kernel_size = kernel_size_,
          weight_filler = dict(type = "gaussian", std = weight_std),
          bias_filler = dict(type = "constant", value = bias_value))
    }

    conv_name = out_layer
    net[conv_name] = L.Convolution(net[from_layer], **kwargs)

    num = conv_name[-1]
    return CRelu(net, conv_name, num)



#CRelu_layer From FaceBox
def CRelu(net, from_layer, from_data):
    batchNorm_name = 'batchNorm{}'.format(from_data)
    net[batchNorm_name] = L.BatchNorm(net[from_layer], batch_norm_param = dict(use_global_stats = False))

    scale_name = 'scale{}'.format(from_data)
    net[scale_name] = L.Scale(net[batchNorm_name], filler = dict(type ='constant', value = -1.0 ))

    concat_name = 'concat{}'.format(from_data)
    net[concat_name] = L.Concat(net[scale_name], net[batchNorm_name])

    relu_name = 'relu{}'.format(from_data)
    net[relu_name] = L.ReLU(net[concat_name],in_place=True )

    return concat_name


#full_conection layer
def full_conection(bottom, lr_mult1, decay_mult1, lr_mult2, decay_mult2,
                num_output_, weight_std, bias_value):
    kwargs = {
    'param':[dict(lr_mult = lr_mult1, decay_mult = decay_mult1),
            dict(lr_mult = lr_mult2, decay_mult = decay_mult2)],
    'inner_product_param': dict(num_output= num_output_,
                     weight_filler =  dict(type = "gaussian",std = weight_std),
                    bias_filler = dict(type = "constant",  value = bias_value))
    }
    fc = L.InnerProduct(bottom, **kwargs)
    return fc

#affine data layer
def AffineDataLayer(image_folder_, landmark_file_, label_file_, batch_size_,
                     center_ind_, norm_ratio_, height_, width_ ):

    kwargs = {
        'multilabel_param': dict(
                                image_folder = image_folder_,
                                landmark_file = landmark_file_,
                                label_file = label_file_,
                                key_point_count = 5,
                                batchsize = batch_size_,
                                affine_image_param = dict(
                                    center_ind = center_ind_,
                                    norm_mode = caffe_pb2.AffineImage_Norm_Mode.Value('RECT_LE_RE_LM_RM'),
                                    norm_ratio = norm_ratio_,
                                    fill_type = False,
                                    value = 0,
                                        image_info = dict(height = height_,
                                        width = width_,
                                        is_color = True
                                        )
                                    )
                                )
                }
    data, label = L.AffineData(ntop=2, **kwargs)
    return [data, label]

#lmdb data layer
def LmdbDataLayer(lmdb_source_path, batch_size_):
    kwargs = {
            'data_param' : dict(source = lmdb_source_path,
                                batch_size = batch_size_,
                                backend = caffe_pb2.DataParameter.DB.Value('LMDB')
                                  )
                       }
    data, label = L.Data(ntop=2, **kwargs)
    return [data, label]

def ImageDataLayer(source_, batch_size_, root_folder_):
    kwargs = {
    'image_data_param' : dict(source = source_,
                        batch_size = batch_size_,
                        root_folder = root_folder_,
                        shuffle = True
                          )
               } 
    data, label = L.ImageData(ntop=2, **kwargs)
    return [data, label]

def ImageDataLayer_Means(source_, batch_size_, root_folder_, means_, scale_):
    kwargs = {
    'image_data_param' : dict(source = source_,
                        batch_size = batch_size_,
                        root_folder = root_folder_,
                        shuffle = True,
                          ),
    
        'transform_param' : dict(
        mean_value = means_,
        scale =  scale_,
        )
               } 


    data, label = L.ImageData(ntop=2, **kwargs)
    return [data, label]


#sphere face <A-softmax Loss>
def A_softmaxLoss(net, from_layer, layer_name, num_out, mode_type, lambda_val, base_num=0):

    A_softmaxLoss_type = {
        1: caffe_pb2.MarginInnerProductParameter.MarginType.Value('SINGLE'),
        2: caffe_pb2.MarginInnerProductParameter.MarginType.Value('DOUBLE'),
        3: caffe_pb2.MarginInnerProductParameter.MarginType.Value('TRIPLE'),
        4: caffe_pb2.MarginInnerProductParameter.MarginType.Value('QUADRUPLE'),
    }

    kwargs = {
        'margin_inner_product_param': dict(
            num_output = num_out,
            type = A_softmaxLoss_type[mode_type],
             weight_filler =  dict(type = "xavier"),
             base = base_num,
             gamma = 0.12,
             power = 1,
             lambda_min = lambda_val,
             iteration = 0,
        )
    }
    net[layer_name], net['lambda'] = L.MarginInnerProduct(net[from_layer], net['label'], ntop=2, **kwargs)
    net['softmax_loss'] = L.SoftmaxWithLoss(net[layer_name], net['label'])

#sphere Auglar TripletLoss loss
def Auglar_tripletLoss(net, from_layer, layer_name, num_out, mode_type, lambda_val, base_num=0, triplet_ = True, semihard_ = True):
    A_softmaxLoss_type = {
        1: caffe_pb2.MarginInnerProductParameter.MarginType.Value('SINGLE'),
        2: caffe_pb2.MarginInnerProductParameter.MarginType.Value('DOUBLE'),
        3: caffe_pb2.MarginInnerProductParameter.MarginType.Value('TRIPLE'),
        4: caffe_pb2.MarginInnerProductParameter.MarginType.Value('QUADRUPLE'),
    }

    kwargs = {
        'margin_inner_product_param': dict(
            num_output = num_out,
            type = A_softmaxLoss_type[mode_type],
             weight_filler =  dict(type = "xavier"),
             base = base_num,
             gamma = 0.12,
             power = 1,
             lambda_min = lambda_val,
             iteration = 0,
             triplet = triplet_,
             semihard = semihard_
        )
    }
    net[layer_name], net['lambda'] = L.MarginInnerProduct(net[from_layer], net['label'], ntop=2, **kwargs)
    net['softmax_loss'] = L.SoftmaxWithLoss(net[layer_name], net['label'])

#sphere Additive-Margin_Loss
def AM_softmaxLoss(net, from_layer, layer_name, num_out, bias_, scale_value):
    net["norm1"] = L.Normalize(net[from_layer])
    name = '{}_l2'.format(layer_name)
    kwargs1 = {
        'param':dict(lr_mult = 1),
        'inner_product_param':dict(
                num_output = num_out,
                normalize = True,
                weight_filler =  dict(type = "xavier"),
                bias_term = False,
        )
    }
    net[name] = L.InnerProduct(net["norm1"], **kwargs1)

    name2 = "{}_margin".format(layer_name)
    net[name2] = L.LabelSpecificAdd(net[name], net['label'], label_specific_add_param = dict(bias = bias_))
    name3 = "{}_margin_scale".format(layer_name)
    kwargs2 = {
        'param':dict(lr_mult = 0, decay_mult = 0),
        'scale_param': dict(
                filler = dict(type = "constant", value = 30)
        )
    }
    net[name3] = L.Scale(net[name2], **kwargs2)
    net['softmax_loss'] = L.SoftmaxWithLoss(net[name3], net['label'])


def softmaxLoss(net, from_layer, layer_name, num_out):
    kwargs1 = {
        'param':[dict(lr_mult = 1.0, decay_mult=1.0),dict(lr_mult = 2.0, decay_mult=0.0)],
        'inner_product_param':dict(
                num_output = num_out,
                weight_filler =  dict(type = "gaussian", std = 0.01),
                bias_filler = dict(type = "constant", value = 0.0),
        )
    }
    net[layer_name] = L.InnerProduct(net[from_layer], **kwargs1)

    net['softmax_loss'] = L.SoftmaxWithLoss(net[layer_name], net['label'])

def softmax(net, from_layer, layer_name, num_out):
    kwargs1 = {
        'param':[dict(lr_mult = 1.0, decay_mult=1.0),dict(lr_mult = 2.0, decay_mult=0.0)],
        'inner_product_param':dict(
                num_output = num_out,
                weight_filler =  dict(type = "gaussian", std = 0.01),
                bias_filler = dict(type = "constant", value = 0.0),
        )
    }
    net[layer_name] = L.InnerProduct(net[from_layer], **kwargs1)

    net['softmax'] = L.Softmax(net[layer_name])

def  TripletRankHardLoss(net, from_layer, neg_num_, hard_ratio_, rand_ratio_, margin_, dist_mode_):
        dist_type = {
        0: caffe_pb2.RankHardLossParameter.DISTANCE_MODE.Value('ELEM_PRODUCT'),
        1: caffe_pb2.RankHardLossParameter.DISTANCE_MODE.Value('L1_DIST'),
        2: caffe_pb2.RankHardLossParameter.DISTANCE_MODE.Value('L2_DIST'),
    }
        kwargs1 = {
            'rank_hard_loss_param': dict(
                neg_num = neg_num_,
                pair_size = 2,
                hard_ratio = hard_ratio_,
                rand_ratio = rand_ratio_,
                margin = margin_,
                dist_mode = dist_type[dist_mode_]
                )
              }
        net['triplet_loss'] = L.RankHardLoss( net[from_layer], net['label'] ,**kwargs1)


def TripletImageDataLayer(source_, batch_size_, root_folder_, pair_size_):

    kwargs = {
    'image_data_param' : dict(source = source_,
                        batch_size = batch_size_,
                        root_folder = root_folder_,
                        shuffle = True,
                        pair_size = pair_size_
                          )
               } 
    data, label = L.TripletImageData(ntop=2, **kwargs)
    return [data, label]

#triplet data layer
def TripletDataLayer(source_, batch_size_, root_folder_, subjects_per_iter_, samples_per_subject_, o3_samples_per_subject_):
        kwargs = {
    'image_data_param' : dict(source = source_,
                        batch_size = batch_size_,
                        root_folder = root_folder_,
                        shuffle = False,
                       subjects_per_iter = subjects_per_iter_,
                       samples_per_subject = samples_per_subject_,
                       o3_samples_per_subject = o3_samples_per_subject_
                          )
               }        
        data, label = L.TripletData(ntop =2, **kwargs )
        return [data, label]