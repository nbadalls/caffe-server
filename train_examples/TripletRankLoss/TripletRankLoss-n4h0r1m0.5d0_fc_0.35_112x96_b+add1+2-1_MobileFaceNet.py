#use python script to create trainNet
#Special for Additive Margin face loss
#Author Zkx@__@
#Date 2017-12-19
#Update 2018-02-08
#param info:
# n4h0r1m0.1d0 -- NegNum4- hard_ratio0- rand_ratio1- margin0.1 dist_mode(0-elem_product 1-l1 2-l2)

from __future__ import print_function
import sys
sys.path.append('./python')
sys.path.append('./python/caffe')
import caffe
from google.protobuf import text_format
#from net_libs import *
from net_function_libs import *

import time
import shutil
import subprocess
import stat
from getPatchInfoFunc import *
from utility import *
import train_set
import pdb
def train_model(device_id, resume_training = None, model_date = None, iterate_num = None):


    py_file_name = os.path.abspath(__file__).split('/')[-1]
    dataset = os.path.splitext(py_file_name)[0].split('_')[-2]

#stepvalue: 80000
#stepvalue: 120000
#stepvalue: 140000
#max_iter:  160000

    run_soon = True
    #solver params
    solver_param = {
        'base_lr': 0.0001,
        'lr_policy': "step",
        #'lr_policy': "fixed",
        #'lr_policy': "multistep",
        'stepsize': 100000,
        'gamma': 0.1,
       
        #'stepvalue': [ 80000, 120000, 160000],
        #'stepvalue': [ 300000],
        'max_iter': 200000,

        'snapshot': 5000, 
    	# 'device_id' : 4,

        'weight_decay': 5e-4,
        'momentum': 0.9,
        'iter_size': 1, #iter_size*batch_size actual
        'solver_type': caffe_pb2.SolverParameter.SolverType.Value('SGD'),
        'solver_mode':caffe_pb2.SolverParameter.SolverMode.Value('GPU'),
        'display': 10,
        'snapshot_after_train':True, #save model after training finished!!
        }
    batch_size_ = 128
    #current time as defult
    best_model_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    #best_model_date = '2018-03-29'
    #split patch param according file name
    # py_file_name = os.path.abspath(__file__).split('/')[-1]
    file_basename = os.path.splitext(py_file_name)[0] #FakeFace_fc_0.3_75x60_DeepID
    prepart_split = file_basename.split('x')[0]

    center_id_part = prepart_split.split('_')[1:-2]
    ratio = prepart_split.split('_')[-2]
    h_value = prepart_split.split('_')[-1]
    w_value = file_basename.split('x')[1].split('_')[0]

    center_ind_ = stringTopatchID(center_id_part)
    norm_ratio_ = float(ratio)
    height_ = int(h_value)
    width_ = int(w_value)

    print(center_ind_)
    print(norm_ratio_)
    print(height_)
    print(width_)
    
    #patch params
    patch_center = patchIDMatching(center_ind_)  #le_lm
    crop_patch = '{}_{}_{}x{}'.format(patch_center, norm_ratio_, height_, width_)
    source_ = train_set.train_mtcnn_img_data[dataset][0] 
    num_out = train_set.train_mtcnn_img_data[dataset][1] 
    root_folder_ = "{}/{}/".format(train_set.image_mtcnn_root_folder,  crop_patch)
    

    train_subject = file_basename.split('_')[0] #FakeFace
    loss_type = train_subject.split('-')[0]
    train_net = file_basename.split('_')[-1] #DeepID
    trainer = "zkx"

    #set triplet loss param  n4h0r1m0.1d0  neg_num_, hard_ratio_, rand_ratio_, margin_, dist_mode_
    
    triplet_param_part = train_subject.split('-')[-1]
    neg_num_index = triplet_param_part.find('n')
    hard_ratio_index = triplet_param_part.find('h')
    rand_ratio_index = triplet_param_part.find('r')
    margin_index = triplet_param_part.find('m')
    dist_mode_index = triplet_param_part.find('d')
    
    neg_num_ = int(triplet_param_part[neg_num_index+1:hard_ratio_index])
    hard_ratio_ = float(triplet_param_part[hard_ratio_index+1:rand_ratio_index])
    rand_ratio_ = float(triplet_param_part[rand_ratio_index+1:margin_index])
    margin_ = float(triplet_param_part[margin_index+1:dist_mode_index])
    dist_mode_ = int(triplet_param_part[dist_mode_index+1:])

  
    print('neg_num_ {} hard_ratio_ {} rand_ratio_ {} margin_ {} dist_mode_index {}'.format(neg_num_, hard_ratio_, rand_ratio_, margin_, dist_mode_))

    #find date model in history best result 
    model_pretraind_path = "/home/zkx/Project/O2N/best_select_models_caffe-master/2018-05-29/2018-05-29_AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn_zkx_iter_150000/2018-05-29_AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn_zkx_iter_150000.caffemodel"
    #model_pretraind_path = None
    best_result_model_path = '../best_select_models/{}/XCH-Ad/{}/{}'.format(best_model_date, train_subject.split('-')[0], file_basename)
    print (best_result_model_path)
    for best_root_path, best_folder,best_filename in os.walk(best_result_model_path):
             for each_filename in best_filename:
                       if each_filename.find('.caffemodel') >=0 :
                               model_pretraind_path = '{}/{}'.format(best_root_path, each_filename)


    #get deploy file name 
    root_deploy_path = './deploy_lib'
    deploy_lib_file = '{}/{}_deploy.prototxt'.format(root_deploy_path, train_net)
    if not os.path.exists(deploy_lib_file):
            print('{} file does not exist!!'.format(deploy_lib_file))
            return 
    #=========================================================================================

    patch_info = "{}_{}_{}_{}".format(train_subject, crop_patch, dataset, train_net) #ImageQulityEvaluation_fc_0.4_100x100_DeepID
    # Modify the job name if you want.
    job_name = patch_info
    # The name of the model. Modify it if you want.
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))  #2017-08-11_ImageQulityEvaluation_fc_0.8_64x64_DeepID_zkx_iter_176000
    model_name = "{}_{}_{}".format(date, job_name, trainer)

    # Directory which stores the model .prototxt file.
    save_dir = "models/{}/{}".format(loss_type, job_name) #asset/ImageQulityEvaluation/ImageQulityEvaluation_fc_0.8_64x64_DeepID
    # Directory which stores the snapshot of models.
    snapshot_dir = "../asset/snapshot/{}/{}".format(loss_type,job_name)
    # Directory which stores the job script and log file.
    job_dir = "jobs/{}/{}".format(loss_type, job_name)

    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    #pretrain_model = '{}/{}.caffemodel'.format(snapshot_dir, '2017-08-16_ImageQulityEvaluation_le_re_0.5_50x80_DeepID_zkx_iter_140000')

    # model definition files.
    train_net_file = "{}/train.prototxt".format(save_dir)
    deploy_net_file = "{}/deploy.prototxt".format(save_dir)
    solver_file = "{}/solver.prototxt".format(save_dir)

    # snapshot prefix.
    snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
    # job script path.
    job_file = "{}/{}.sh".format(job_dir, model_name)


    ### Hopefully you don't need to change the following ###
    # Check file.
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)
    make_if_not_exist(snapshot_dir)


    #=================================Create Date layer======================
    data_layer = caffe.NetSpec()
    data_layer['data'], data_layer['label']  = TripletImageDataLayer(source_, batch_size_, root_folder_, 2)

    #create net body
    net_proto_layer = caffe_pb2.NetParameter()
    #delete the normlize layer and slience layer
    net_body_proto = read_deploy_into_proto_config_delete_num(deploy_lib_file, net_proto_layer, False, 1)
    last_layer_name = net_body_proto.layer[-1].name
    #when appearance bn layer in the end
    #if last_layer_name.find("_bn") >=0 :
    #    last_layer_name = last_layer_name.split("_")[0]
    
    #layer_num = int(last_layer_name[-2:])
    #layer_name = 'fc{}'.format(layer_num + 1 )
    
   
    
    #Create loss layer
    loss_layer = caffe.NetSpec()
    loss_layer[last_layer_name], loss_layer['label'] = L.ImageDataParameter(ntop =2)
    
    TripletRankHardLoss(loss_layer, last_layer_name, neg_num_, hard_ratio_, rand_ratio_, margin_, dist_mode_)
    loss_layer_proto = loss_layer.to_proto()
    #remove the ImageDataParameter layer
    del loss_layer_proto.layer[0]

    #save into train_val
    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(data_layer.to_proto(), file=f)
        print(net_body_proto, file = f)
        print(loss_layer_proto, file = f)
    shutil.copy(train_net_file, job_dir)

#=================================Create Date layer================================================

    # Create deploy net.
    shutil.copy(deploy_lib_file, deploy_net_file)
    #modifiy input of deploy
    net_proto = caffe_pb2.NetParameter()
    f = open(deploy_net_file, 'r')
    text_format.Merge(f.read(), net_proto)
    f.close()
    net_proto .input_dim[0]  = 1
    net_proto .input_dim[1]  = 3
    net_proto .input_dim[2]  = height_
    net_proto .input_dim[3]  = width_

    f  = open(deploy_net_file, 'w')
    print(net_proto, file = f)
    f.close()
    #copy to job dir
    shutil.copy(deploy_net_file, job_dir)

    # Create solver.
    solver = caffe_pb2.SolverParameter(
            net=train_net_file,
           # deploy_net=deploy_net_file,
            snapshot_prefix=snapshot_prefix,
            **solver_param)
    with open(solver_file, 'w') as f:
        print(solver, file=f)


    if resume_training != None:
        #find most recent date
        max_year, max_month, max_day = find_most_recent_date(snapshot_dir)
        most_recent_date = '{}-{}-{}'.format(max_year, str(max_month).zfill(2), str(max_day).zfill(2))
        #find max iteration number on most recent date
        max_iter = find_biggest_iter_num(snapshot_dir, most_recent_date)

        if model_date != None:
                most_recent_date =  model_date
        if iterate_num != None:
                max_iter = iterate_num

        train_src_param = ""
        if max_iter > 0:
            if resume_training:
        	       train_src_param = '--snapshot="{}/{}_{}_{}_iter_{}.solverstate" \\\n'.format(snapshot_dir,most_recent_date, patch_info, trainer, max_iter )
            else:
        	       train_src_param = '--weights="{}/{}_{}_{}_iter_{}.caffemodel" \\\n'.format(snapshot_dir,most_recent_date, patch_info, trainer, max_iter )
    elif model_pretraind_path != None:
            train_src_param = '--weights="{}" \\\n'.format(model_pretraind_path)
    else:
             train_src_param = ''

    # Create job file.
    caffe_root = os.getcwd()
    with open(job_file, 'w') as f:
      f.write('cd {}\n'.format(caffe_root))
      f.write('./build/tools/caffe train \\\n')
      f.write('--solver="{}" \\\n'.format(solver_file))

    #gpu device divide
      split_device = ""
      for elem in device_id:
            split_device+='{},'.format(elem)
      split_device = split_device.strip(',')
      f.write('--gpu="{}" \\\n'.format(split_device))
      f.write(train_src_param)


      if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('2>&1 | tee {}/{}.log\n'.format( job_dir, model_name))
      else:
        f.write('2>&1 | tee {}/{}.log\n'.format( job_dir, model_name))



    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, job_dir)

    # Run the job.
    os.chmod(job_file, stat.S_IRWXU)
    if run_soon:
      subprocess.call(job_file, shell=True)

if __name__ == '__main__':

    #train with max iteration number and latest date's .caffemodel if resume_training == False
    #train with max iteration number and latest date's .solverstate if resume_training == True
    #if(len(sys.argv) == 2):
        #if sys.argv[1] == 'true':
        #    resume_training = True
        #elif sys.argv[1] == 'false':
        #    resume_training = False

        #train_model(resume_training)
    #elif(len(sys.argv)== 4):
     #   if sys.argv[1] == 'true':
      #      resume_training = True
       # elif sys.argv[1] == 'false':
        #    resume_training = False

        #model_date = sys.argv[2]
        #iterate_num = sys.argv[3]
        #train_model(resume_training, model_date, iterate_num)

    #elif(len(sys.argv)== 5):
        #if sys.argv[1] == 'true':
         #   resume_training = True
        #elif sys.argv[1] == 'false':
        #    resume_training = False

        #model_date = sys.argv[2]
        #iterate_num = sys.argv[3]
        #model_pretraind_path = sys.argv[4]
        #train_model(resume_training, model_date, iterate_num, model_pretraind_path)
    if len(sys.argv) == 2:
          device_id = sys.argv[1]
          train_model(device_id)
    elif len(sys.argv) == 3:
            device_id = sys.argv[1]
            if sys.argv[2] == 'true':
                resume_training = True
            elif sys.argv[2] == 'false':
                resume_training = False
            train_model(device_id, resume_training)
    else:
        print ("Input: param -resume_training (true, or false), -model_date = None, -iterate_num = None -model_pretraind_path = None")
