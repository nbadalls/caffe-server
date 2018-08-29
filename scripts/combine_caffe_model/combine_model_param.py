# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:01:52 2018

@author: minivision
"""
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
from google.protobuf import text_format
import combine_utility
import copyLayersFromMultipleSingleNetsToOneModel

class combineModelParam():

    def __init__(self, single_root_path, dst_combine_path):
        self.single_root_path = single_root_path
        self.dst_combine_path = dst_combine_path

    #patch_folder = ['FakeFace_fc_0.4_96x96_DeepID_S', 'FakeFace_le_0.3_80x80_DeepID', 'FakeFace_le_re_n_0.8_60x60_DeepID']
    #prefix_name = ['fc_0.4_96X96', 'le_0.3_80X80', 'le_re_n_0.8_60X60']
    def config_net_path(self, patch_folder, prefix_names):
        nets, dst_model_path = combine_utility.create_single_net(self.single_root_path, self.dst_combine_path, patch_folder, prefix_names)
        self.nets = nets
        self.dst_model_path = dst_model_path
        self.prefix_names = prefix_names

#sinple version simplely combine net together
    def create_combine_deploy(self ):
        print("Father class")
        net_proto , record_layer_index= combine_utility.combine_single_deploy(self.nets, 0)
        print(record_layer_index)
        f = open(self.dst_model_path['dst_deploy'], 'w')
        print(net_proto, file = f)
        f.close()

    def create_prefix_deploy(self):
                 combine_utility.create_single_prefix_deploy(self.nets)

    def copy_params_to_combine_model(self, prefix_names):
            joint_net_params = {}
            joint_net_params['net_configuration'] = self.dst_model_path['dst_deploy']
            joint_net_params['output_model'] = self.dst_model_path['dst_model']

            single_nets_params = {}
            for i in range(len(prefix_names)):
                single_nets_params[prefix_names[i]] = self.nets[i]

            copyLayersFromMultipleSingleNetsToOneModel.CombineCaffeModels(joint_net_params, single_nets_params)


#run scripts...
    def model_combination(self, patch_folder, prefix_names):
        self.config_net_path(patch_folder, prefix_names)
        #create prefix deploy and matched list
        self.create_prefix_deploy()
        #create combined deploy
        self.create_combine_deploy()
        #copy single net param into combined model according to related match
        self.copy_params_to_combine_model(prefix_names)



if __name__ == '__main__':
    root_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/HistoryBestModel"
    dst_path = "/media/minivision/OliverSSD/LiveBody/select_best_result/"
    patch_folder = ["2018-08-14_AgeGenderMtcnn_fc_0.25_90x80_age-gender-dataset1_DeepID_zkx_iter_105000",
    "2018-08-15_AgeGenderMtcnn_fc_0.25_90x80_age-gender-dataset1_MobileFaceNet-k-5-5_zkx_iter_50000" ]
    prefix_names = ["student", "teacher"]
    combine_model = combineModelParam(root_path,dst_path )
    combine_model.model_combination(patch_folder, prefix_names)
