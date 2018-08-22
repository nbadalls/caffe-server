from __future__ import print_function
import sys
sys.path.append('./python')
sys.path.append('./python/caffe')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format




class ModifyDeploy():

	def __init__(self,src_deploy_path, dst_deploy_path):
		self.src_deploy_path = src_deploy_path
		self.dst_deploy_path = dst_deploy_path


	def read_deploy_file(self, deploy_path):
		net_proto = caffe_pb2.NetParameter()
		f = open(deploy_path, 'r')
		text_format.Merge(f.read(), net_proto)
		f.close()
		return net_proto

	def write_deploy_file(self, net_proto, deploy_path):
		with open(deploy_path, 'w') as f:
			print(net_proto, file = f)

	def change_convolution_to_depthwiseConvolution(self):

		net_proto = self.read_deploy_file(self.src_deploy_path)
		for elem_layer in net_proto.layer:
			if elem_layer.name.find('dwise') >=0 and elem_layer.type == "Convolution":
				elem_layer.type == "DepthwiseConvolution"

		self.write_deploy_file(net_proto, self.dst_deploy_path)



if __name__ == '__main__':

	src_deploy_path = "/home/zkx-97/Project/O2N/caffe-master/deploy_lib/mobilenet-v2_deploy.prototxt"
	dst_deploy_path = "/home/zkx-97/Project/O2N/caffe-master/modify_deploy/mobilenet-v2-dp_deploy.prototxt"
	mod_deploy = ModifyDeploy(src_deploy_path, dst_deploy_path)
	mod_deploy.change_convolution_to_depthwiseConvolution()