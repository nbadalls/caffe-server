#use python script to create trainNet
#Special for Additive Margin face loss
#Author Zkx@__@
#Date 2017-12-19
#Update 2018-02-08
from __future__ import print_function
import sys
sys.path.append('./python')
sys.path.append('./python/caffe')
import caffe
from google.protobuf import text_format
# from net_libs import *
from net_function_libs import *

import time
import shutil
import subprocess
import stat
from getPatchInfoFunc import *
from utility import *
import train_set
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
        #'lr_policy': "step",
        'lr_policy': "fixed",
        #'lr_policy': "multistep",
        #'stepsize': 150000,
        'gamma': 0.1,
       
        #'stepvalue': [ 15000, 30000, 100000,],
        #'stepvalue': [ 300000],
        'max_iter': 300000,

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
    source_ = train_set.age_gender_image_data[dataset]
    age_num_out = 6
    gender_num_out = 2 

    root_folder_ = "{}/{}/".format(train_set.age_gender_image_root_path,  crop_patch)
    # lmdb_source_path = "/home/zkx/Data_sdb/TrainData/Data_sdc/Patches/shot_Oriental_Age_Lan_DHUA_PAKJ_Indon_Migrant_celeb/fc_0.35_112x96-train_cvdecode-shuffle_lmdb"
    #train_image_folder_ = "/home/zkx/Data_sdb/TrainData"
    #train_landmark_file_ = "/home/zkx/Data_sdb/TrainData_landmarks/combine_folder_landmark/Combine_shot_Orien_Age_LanDH_PAKJ_Ind_Mig_Msceleb_AsLife_O2O-6000_landmarks_deplicate.txt"
    #train_label_file_ = "/home/zkx/Data_sdb/TrainData_landmarks/combine_folder_landmark/Combine_shot_Orien_Age_LanDH_PAKJ_Ind_Mig_Msceleb_AsLife_O2O-6000_landmarks_deplicate_label.txt"
    

    train_subject = file_basename.split('_')[0] #FakeFace
    loss_type = train_subject.split('-')[0]
    train_net = file_basename.split('_')[-1] #DeepID
    trainer = "zkx"

    #set AM face loss param  bias, scale_value
    
    # sphere_param_part = train_subject.split('-')[-1]
    # b_index = sphere_param_part.find('b')
    # s_index = sphere_param_part.find('s')
    # bias = -float(sphere_param_part[b_index+1:s_index])
    # scale_value = float(sphere_param_part[s_index+1:])
    
    
    # print('b{}s{}'.format(bias, scale_value))

    #find date model in history best result 
    model_pretraind_path = None
    best_result_model_path = '../train_models/best_select_models/{}/XCH-Ad/{}/{}'.format(best_model_date, train_subject.split('-')[0], file_basename)
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

    snapshot_dir = "../../models/AgeGender/asset/snapshot/{}/{}".format(loss_type,job_name)
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
    print(source_, batch_size_, root_folder_)
    data_layer['data'], data_layer['label']  = ImageMultilabelDataLayer(source_, batch_size_, root_folder_)
    # data_layer['data'], data_layer['label'] = LmdbDataLayer(lmdb_source_path, batch_size)

    #label rank first age  -- second gender
    data_layer['age_label'], data_layer['gender_label'] = L.Slice(data_layer['label'] , slice_param = dict(axis = 1, slice_point=1), 
                                                                                            ntop=2, name = 'label_slice')



    #create net body
    net_proto_layer = caffe_pb2.NetParameter()
    #delete the normlize layer and slience layer
    net_body_proto = read_deploy_into_proto_delete_slience_norm(deploy_lib_file, net_proto_layer, False)
    last_layer_name = net_body_proto.layer[-1].top[0]
    #when appearance bn layer in the end
    if last_layer_name.find("_bn") >=0 :
        last_layer_name = last_layer_name.split("_")[0]
    
    layer_num = int(last_layer_name[-1])
    layer_name = 'fc{}'.format(layer_num + 1 )
    
   
    
    #Create loss layer
    loss_layer = caffe.NetSpec()
    loss_layer[last_layer_name], loss_layer['age_label'] , loss_layer['gender_label']= L.ImageData(ntop =3)

    age_layer_name = '{}-age-'.format(layer_name)
    gender_layer_name = '{}-gender'.format(layer_name)

    softmaxIgnore(loss_layer, last_layer_name, age_layer_name, "age_label", age_num_out, -1)
    softmaxIgnore(loss_layer, last_layer_name, gender_layer_name, "gender_label", gender_num_out, -1)

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
    #change train val into deploy file
    softmax_layer = caffe.NetSpec()
    softmax_layer[last_layer_name]= L.ImageDataParameter()
    #add softmax layer
    softmax(softmax_layer, last_layer_name, age_layer_name, age_num_out)
    softmax(softmax_layer, last_layer_name, gender_layer_name, gender_num_out)
    
    softmax_layer_proto = softmax_layer.to_proto()
    del softmax_layer_proto.layer[0]

    #change batchnorm using global stats back to true
    for elem in net_body_proto.layer:
        if elem.type == 'BatchNorm':
            elem.batch_norm_param.use_global_stats = True
    #add data shape
    net_body_proto.name = model_name
    net_body_proto.input.extend(['data'])
    net_body_proto.input_dim.extend([1,3, height_, width_])
    #write into file
    with open(deploy_net_file, 'w') as f:
        print(net_body_proto, file=f)
        print(softmax_layer_proto, file=f)

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
