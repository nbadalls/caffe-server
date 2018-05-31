# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:49:45 2018
1. Based on the face_verification_test result
2. Choose best accuracy model according to result log file
3. Select best accuracy model to ConfigPath.best_model_path     findBestModel() [1-3]
4. Select face verification model beyond the threshold          selectVerificationModel(self, threshold)[4]


@author: zkx-96
"""

import os
import time
import shutil

from config_path import *
import utility

class ModelSelect():
    
    def __init__(self, current_date, test_set_type):
       
        self.test_set_type = test_set_type
        self.current_date = current_date
        
        #where store the test result
        self.test_result_root_path = '{}/{}/{}'.format(ConfigPath.out_root_path, current_date, test_set_type)
        
        
    def findBestModel(self):
            out_model_path = '{}/{}/{}'.format(ConfigPath.best_model_path, self.current_date, self.test_set_type)
            best_model_acc_pair = []
            test_result_path = '{}/{}/{}'.format(ConfigPath.out_root_path, self.current_date, self.test_set_type)
            print(test_result_path)
            for root_path, folder_path, filename_path in os.walk(test_result_path):
                for filename in filename_path:
                    if filename.find('roc_statistic_result') >=0 and filename.endswith('log'):
                        log_path = '{}/{}'.format(root_path, filename)
                        best_pair = self.parseLogFile(log_path)
                        best_model_acc_pair += best_pair
                        
            #rank accord to model name
            best_model_acc_pair.sort(key = lambda k:k[0])            
            #save into file
            if len(best_model_acc_pair) >0:
                utility.make_dirs(out_model_path)
                date_minu = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                best_model_list = '{}/statistic_info_{}.log'.format(out_model_path, date_minu)
                f = open(best_model_list, 'w')
                for elem in best_model_acc_pair:
                    #print (elem)
                    f.write('{} {}\n'.format(elem[0], elem[1]))
                f.close()
                
            #select models
            self.selectBestModel(best_model_acc_pair, out_model_path)
            
            
    #select models beyond threshold            
    def selectVerificationModel(self, threshold):
        best_log_path = '{}/{}/{}'.format(ConfigPath.best_model_path, self.current_date, self.test_set_type)
        log_file_list = [elem for elem in os.listdir(best_log_path) if elem.endswith('.log')]
        
        #read informations in the log file
        best_model_acc_pair = {}
        for log_file in log_file_list:
            f = open('{}/{}'.format(best_log_path, log_file), 'r')
            data = f.read().splitlines()
            f.close()
            for line in data:
                parts = line.split(' ')
                if parts[0] not in best_model_acc_pair.keys():
                    best_model_acc_pair[parts[0]] = parts[1]
            
        best_verification_path = '{}/{}/{}'.format(ConfigPath.best_verification_model_path, self.current_date, self.test_set_type)
        utility.make_dirs(best_verification_path)
        self.selectBestModel(best_model_acc_pair.items(), \
                                    best_verification_path, \
                                    threshold)
            
            
    #parse result log file to find best accuracy model each Net        
    def parseLogFile(self, log_file):
        f = open(log_file, 'r')
        data = f.read().splitlines()
        f.close()
 
        if len(data) == 0:
            return []
       
        static_dict = {}
        model_name = ""
        accuracy = "fpr	0.00012	acc"
        for line in data:
              if line.find('result_v2') >=0 :
                  model_name = line.split('result_v2_')[-1].strip('.txt')
              if line.find(accuracy) >= 0:               #fpr	0.00012 acc	0.69776	threshlod	1.00739
                  #print(line.split('\t'))
                  acc_value = float(line.split('\t')[3])
                  static_dict[model_name] = acc_value
                 
        #sorted value asscend
        dict_item = static_dict.items()
        dict_item.sort(key = lambda k:k[1])
        
        #select best two models
        return dict_item[-3:-1]  #{model name -- accuracy}
            
    def selectBestModel(self, best_model_acc_pair, out_model_path, threshold = 0.0):
        
        #sort according to model name
        best_model_acc_pair.sort(key = lambda k:k[0])  
        for elem in best_model_acc_pair:
            model_name = elem[0]
            model_acc = float(elem[1])
            
            if model_acc >= threshold:
                print("{}\t{}".format(model_name, model_acc))
                train_date = model_name.split('_')[0]
                trainer = model_name.split('_')[-3]
                end_sign = model_name.find(trainer)-1
                
                model_prefix = model_name[0:end_sign].split(train_date + '_')[1]
                #print(model_prefix)
                loss_type =  model_prefix.split('_')[0]
                index = loss_type.find('-')
                if index >=0:
                    loss_type = loss_type[0:index]
                
                #set caffe model path and test result path deploy path
                model_path = '{}/{}/{}'.format(ConfigPath.model_root_path, loss_type, model_prefix)
                test_result_path = '{}/{}/distance_result'.format(self.test_result_root_path, model_prefix)
                test_deploy_path = '{}/{}/train_file'.format(self.test_result_root_path, model_prefix)
                
                #copy model
                copy_model_path = '{}/{}/models'.format(out_model_path, model_prefix)
                utility.make_dirs(copy_model_path)
                for elem_model_name in os.listdir(model_path):
                    if elem_model_name.find(model_name) >=0 and elem_model_name.endswith('.caffemodel'):
                        copy_model_name = elem_model_name.replace(train_date, self.current_date)
                        shutil.copy('{}/{}'.format(model_path, elem_model_name) , \
                        '{}/{}'.format(copy_model_path, copy_model_name))
                
                #copy roc curve
                for roc_name in os.listdir(test_result_path):
                    if roc_name.endswith('.png') and roc_name.find(model_name) >=0 :
                        copy_roc_name = roc_name.replace(train_date, self.current_date)
                        shutil.copy('{}/{}'.format(test_result_path, roc_name) ,\
                                    '{}/{}'.format(copy_model_path, copy_roc_name))
                                    
                #copy deploy file
                copy_deploy_path = '{}/{}/train_file'.format(out_model_path, model_prefix)
                utility.make_dirs(copy_deploy_path)
                deploy_file = os.listdir(test_deploy_path)[0]
                shutil.copy('{}/{}'.format(test_deploy_path, deploy_file), \
                            '{}/deploy.prototxt'.format(copy_deploy_path))
                        

            
                    
                    
            
                    
            
            
        
                