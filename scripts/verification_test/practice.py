# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:59:26 2018

@author: zkx-96
"""

import time
import sys
import os

import face_verification_test
import select_best_result
from config_path import *


#save_root_path = "/home/zkx/Project/O2N/asset/snapshot/AMImageCdata"
#
#model_folder = ["AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet", "AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn"]
#select_date = "2018-05-30"
#test_set_type = "XCH-Ad"
#gpu_id = 3
#test_batch_num = 20


def modelTest(select_date, test_set_type, gpu_id, test_batch_num = -1):
    
        #init class
    test = face_verification_test.ModelTest(select_date, test_set_type, gpu_id, test_batch_num)
    current_date = test.current_date 
    select = select_best_result.ModelSelect(current_date, test_set_type)
    
    for root_path, folder_path, filename_path in os.walk(ConfigPath.model_root_path):
        for elem in folder_path:      
                model_path = '{}/{}'.format(root_path, elem)            
                test.runTest(model_path)
                
    #after finished all model test then select model
    select.findBestModel()

def modelTestSelect(test_set_type, model_num):
	current_date = time.strftime('%Y-%m-%d', time.localtime())
	select = select_best_result.ModelSelect(current_date, test_set_type)
	select.findBestModel(model_num)

def selectThreshold(test_set_type, threshold):
    current_date = time.strftime('%Y-%m-%d', time.localtime())
    select = select_best_result.ModelSelect(current_date, test_set_type)
    select.selectVerificationModel(threshold)
    

if __name__ == '__main__':

    if len(sys.argv) == 4:
        if sys.argv[1] == "select":
              test_set_type = sys.argv[2]
              threshold = float(sys.argv[3])
              selectThreshold(test_set_type, threshold)
	elif sys.argv[1] == "test-m":    #use to select best model num 
	      test_set_type = sys.argv[2]
	      model_num = int(sys.argv[3])
	      modelTestSelect(test_set_type, model_num)
        else:
              print ("Please input: \n test \n --select_date\n --test_set_type\n --gpu_id\n --test_batch_num[-1]\n")
              print ("select \n --test_set_type\n --threshold\n")
	      print ("test-m \n  --test_set_type\n --model_num\n")
	      
    elif len(sys.argv) == 6:
        if sys.argv[1] == "test":
            select_date = sys.argv[2]
            test_set_type = sys.argv[3]
            gpu_id = int(sys.argv[4])
            test_batch_num = int(sys.argv[5])
            modelTest(select_date, test_set_type, gpu_id, test_batch_num)
        else:
              print ("Please input: \n test \n --select_date\n --test_set_type\n --gpu_id\n --test_batch_num[-1]\n")
              print ("select \n --test_set_type\n --threshold\n")
	      print ("test-m \n  --test_set_type\n --model_num\n")
              
    elif len(sys.argv) == 5:
        if sys.argv[1] == "test":
            select_date = sys.argv[2]
            test_set_type = sys.argv[3]
            gpu_id = int(sys.argv[4])
            modelTest(select_date, test_set_type, gpu_id)
        else:
              print ("Please input: \n test \n --select_date\n --test_set_type\n --gpu_id\n --test_batch_num[-1]\n")
              print ("select \n --test_set_type\n --threshold\n") 
	      print ("test-m \n  --test_set_type\n --model_num\n") 
    else:
              print ("Please input: \n test \n --select_date\n --test_set_type\n --gpu_id\n --test_batch_num[-1]\n")
              print ("select \n --test_set_type\n --threshold\n")   
	      print ("test-m \n --test_set_type\n --model_num\n")       
              

