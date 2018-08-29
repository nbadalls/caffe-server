# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 21:09:09 2016
#Modified by zkx@__@
#2017-10-26
"""

import numpy as np
import sys

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/tools/python
#sys.path.insert(0, caffe_root + 'python')
sys.path.append("/home/minivision/SoftWare/caffe-server/python")
import caffe

def CombineCaffeModels(joint_net_params, single_nets_params):
    # Load the joint network
    joint_net = caffe.Net(joint_net_params['net_configuration'],caffe.TRAIN)


    for single_nets_name, each_single_net_params in single_nets_params.iteritems():
        print ">>>>>>> Copying {} -> joint net <<<<<<<<<<".format(single_nets_name)
        #read the layer map
        with open(each_single_net_params['layer_map'],'r') as f:
            layer_map = { line.split(',')[0]:line.split(',')[1]
                             for line in f.read().splitlines()
                                if len(line.split(',')) > 1
                         }

        single_net = caffe.Net(each_single_net_params['net_configuration'],
                               each_single_net_params['pretrained_model'],caffe.TRAIN)
        #single_net.params.keys()
        for layerName, parameters in single_net.params.iteritems():
            if layerName in layer_map:
                jointLayerName = layer_map[layerName]
                numOfParams = len(parameters)
                for idx in xrange(numOfParams):
                    joint_net.params[jointLayerName][idx].data[:] = parameters[idx].data[:]
                    #print type(parameters[0].data)
                #print len(parameters[0].data)
                print "{} -> {}".format(layerName,jointLayerName)
            else:
                assert False, "Cannot find corresponding layer {} of {} in the joint net".format(layerName,single_nets_name)

    joint_net.save(joint_net_params['output_model'])
    joint_net = []

    ############ Test the save final joint net ##############################################
    new_joint_net = caffe.Net(joint_net_params['net_configuration'],
                          joint_net_params['output_model'],caffe.TRAIN)

    for single_nets_name, each_single_net_params in single_nets_params.iteritems():
        print ">>>>>>> Testing {} in joint net <<<<<<<<<<".format(single_nets_name)
        #read the layer map
        with open(each_single_net_params['layer_map'],'r') as f:
            layer_map = { line.split(',')[0]:line.split(',')[1]
                             for line in f.read().splitlines()
                                if len(line.split(',')) > 1
                         }

        single_net = caffe.Net(each_single_net_params['net_configuration'],
                               each_single_net_params['pretrained_model'],caffe.TRAIN)

        for layerName, parameters in single_net.params.iteritems():
            if layerName in layer_map:
                jointLayerName = layer_map[layerName]
                numOfParams = len(parameters)
                diff = 0
                for idx in xrange(numOfParams):
                    diff += np.sum(new_joint_net.params[jointLayerName][idx].data - parameters[idx].data)
                if diff == 0:
                    print "{} -> {} [TEST PASS]".format(layerName,jointLayerName)
                else:
                    print "{} -> {} [TEST FAILED]".format(layerName,jointLayerName)
            else:
                assert False, "Cannot find corresponding layer {} of {} in the joint net".format(layerName,single_nets_name)
