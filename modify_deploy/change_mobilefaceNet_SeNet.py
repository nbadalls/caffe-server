from __future__ import print_function
import sys
sys.path.append('./python')
sys.path.append('./python/caffe')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def create_SE_part(net, from_layer, conv_num, index, channel_size, concat_layer_name1, concat_layer_name2):

	global_pool_name = "conv{}_{}_global_pool".format(conv_num, index)

	net[global_pool_name] = L.Pooling(net[from_layer], 
		pooling_param = dict(
			pool = caffe_pb2.PoolingParameter.PoolMethod.Value('AVE'), 
 			engine = caffe_pb2.PoolingParameter.Engine.Value("CAFFE"), 
			global_pooling = True))

	down_dim_name = "conv{}_{}_1x1_down".format(conv_num, index)

	#down default as 1/16 of channel numbers of last layer
	net[down_dim_name] = L.Convolution(net[global_pool_name], convolution_param = dict(
		num_output = channel_size/16,  kernel_size = 1, stride = 1) , weight_filler = dict(type = "xavier"))

	down_dim_relu = "{}/relu".format(down_dim_name)
	net[down_dim_relu] = L.ReLU(net[down_dim_name] , in_place=True)

	up_dim_name=  "conv{}_{}_1x1_up".format(conv_num, index)
	net[up_dim_name] = L.Convolution(net[down_dim_relu], convolution_param = dict(
		num_output = channel_size,  kernel_size = 1, stride = 1) , weight_filler = dict(type = "xavier"))	
	up_dim_prob = "conv2_1_prob"
	net[up_dim_prob] = L.Sigmoid(net[up_dim_name] ,in_place=True)

	Axpy_name = 'conv{}_{}'.format(conv_num, index)
	print(up_dim_name, concat_layer_name1, concat_layer_name2)
	net[Axpy_name] = L.Axpy(net[up_dim_name] , net[concat_layer_name1], net[concat_layer_name2])

	return Axpy_name


def add_SE_part_into_MobileFaceNet(net_deploy_path, dst_deploy_path):

     net_proto = caffe_pb2.NetParameter()
    #  net_proto_layer = caffe_pb2.NetParameter()
     f = open(net_deploy_path, 'r')
     text_format.Merge(f.read(), net_proto)
     f.close()


     is_add_net = False
     add_net_top = ""
     SeNet_proto = caffe_pb2.NetParameter()
     #copy original net into SeNet_proto
     for index, elem_layer in enumerate(net_proto.layer):
     	next_index = index +1
     	if next_index > len(net_proto.layer)-1:
     		next_index = len(net_proto.layer)-1
     	#insert SE uint
     	if elem_layer.name.find("em/scale") >=0 and net_proto.layer[next_index].type == "Eltwise":
     		SeNet_proto.layer.extend([elem_layer])

     		#get layer param
     		#conv2_1_em/scale
     		name = elem_layer.name
     		part_name = name.split("_em/scale")[0]
     		conv_num = int(part_name.split('_')[0].split('conv')[1])
     		conv_index = int(part_name.split('_')[1])
     		#get channel number from convolution layer (index-2)
     		channels = int(net_proto.layer[index-2].convolution_param.num_output)
     		cat_layer_name1 = SeNet_proto.layer[-1].top[0]
     		if conv_index == 1:
     			cat_layer_name2 = "conv{}_em".format(conv_num)
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
			
			if is_add_net:
				#change bottom name
				del SeNet_proto.layer[-1].bottom[0]
				SeNet_proto.layer[-1].bottom.extend([add_net_top])
				is_add_net = False

     SeNet_proto.input.extend([net_proto.input[0]])
     SeNet_proto.input_dim .extend([net_proto.input_dim[0], net_proto.input_dim[1], net_proto.input_dim[2], net_proto.input_dim[3]])

     with open(dst_deploy_path, 'w') as f:
	print(SeNet_proto, file = f)


if __name__ == '__main__':
	net_deploy_path = "./deploy_lib/MobileFaceNet-bn_deploy.prototxt"
	dst_deploy_path = "./modify_deploy/deploy.prototxt"
	add_SE_part_into_MobileFaceNet(net_deploy_path, dst_deploy_path)




