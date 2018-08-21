#define training set 

#b(base) -- shot_Oriental_Age_Lan_DHUA_PAKJ_Indon_Migrant_celeb
#FaceAdd -- ZheD-_Cmul-Asia-beid-cap10
#MS -- MSceleb 

image_root_folder = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_haar"
label_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_haar/image_list/train_list"
train_img_data = {
	
	'b+FaceAdd+Mobile-del6k':["{}/Combine_base_Cmul-Asia-beid-cap10-ZheD-CHAMob_list_del2018-03-28_label.txt".format(label_root_path),49742 ],
        'b+FaceAdd+Mobile': ["{}/Combine_base_Cmul-Asia-beid-cap10-ZheD-CHAMob_list.txt_label.txt".format(label_root_path),55918],
	'b+FaceAdd': [ "{}/Combine_base_Cmul-Asia-beid-cap10-ZheD_list.txt_label.txt".format(label_root_path), 54547], 
	'base' :['{}/shot_Oriental_Age_Lan_DHUA_PAKJ_Indon_Migrant_celeb_label.txt'.format(label_root_path), 21331],
}



lmdb_root_path =  "/home/zkx/Data_sdb/TrainData/Data_sdc/Patches/lmdb"
train_lmdb_data = {
	'b+FaceAdd+encode': [ "{}/fc_0.35_112x96_base_ZheD_Cmul-beid_cap10_Asia-train-encode-shufflelmdb".format(lmdb_root_path), 54547],
}


image_mtcnn_root_folder = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn/MultiPatches"
#image_mtcnn_root_folder = "/home/hjg/Data/SDB_Disk/Data/Train_Data/O2N/Data_sdd/Patches_mtcnn/MultiPatches"
label_mtcnn_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn/MultiPatches_list/combine_folder_list"
#label_mtcnn_root_path = "/home/hjg/Data/SDB_Disk/Data/Train_Data/O2N/Data_sdd/Patches_mtcnn/MultiPatches_list/combine_folder_list"
train_img_data_mtcnn = {

        'b+add1+2': ["{}/combine_mtcnn_base+add1-2_GE8_list_label.txt".format(label_mtcnn_root_path),67355],
	'b+add1+2-1': ["{}/combine_mtcnn_base+add1+2-1_list_label.txt".format(label_mtcnn_root_path),69217],
	'b+add1+Chinap2-delAsia-b3': ["{}/combine_mtcnn_base+add1+Chinap2_list_delAsia-b3_label.txt".format(label_mtcnn_root_path),48633],
	'b+add1+Chinap2+GLST-delAsiab3-celeb': ["{}/combine_mtcnn_base+add1+China+GLST_delAsiab3-celeb_list_label.txt".format(label_mtcnn_root_path),78500],
}


clean_mtcnn_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn/MultiPatches_list/clean_combine_list"
train_mtcnn_clean_img_data  = {
      'clean-b+add1+2-1-delAsia-b3-P0.5': ['{}/\
result_face_recognition_predict_2018-07-18_AMImageMtcnn-b0.35s30_\
fc_0.35_112x96_b+add1+2-1-delAsia-b3_MobileFaceNet-bn_zkx_iter_200000_\
correct_possi0.5_select_GE_8_label.txt' .format(clean_mtcnn_root_path), 49099], 

      'clean-b+add1+2-1-delAsia-b3-P0.8': ['{}/\
result_face_recognition_predict_2018-07-18_AMImageMtcnn-b0.35s30_\
fc_0.35_112x96_b+add1+2-1-delAsia-b3_MobileFaceNet-bn_zkx_iter_200000_\
correct_possi0.8_select_GE_8_label.txt' .format(clean_mtcnn_root_path), 48005], 

      'clean-b+add1+2-1-delAsia-b3-P0.9': ['{}/\
result_face_recognition_predict_2018-07-18_AMImageMtcnn-b0.35s30_\
fc_0.35_112x96_b+add1+2-1-delAsia-b3_MobileFaceNet-bn_zkx_iter_200000_\
correct_possi0.9_select_GE_8_label.txt' .format(clean_mtcnn_root_path), 47623], 

      'clean-b+add1+2-1-delAsia-b3-P0.0': ['{}/\
result_face_recognition_predict_2018-07-18_AMImageMtcnn-b0.35s30_\
fc_0.35_112x96_b+add1+2-1-delAsia-b3_MobileFaceNet-bn_zkx_iter_200000_\
correct_possi0.0_select_GE_8_label.txt' .format(clean_mtcnn_root_path), 51464], 
}



final_mtcnn_root_path = "/mnt/glusterfs/o2n/FaceRecognition/Train_Data/O2N_Patches/Patches_mtcnn/MultiPatches_list/combine_folder_combine_add_list"
train_mtcnn_final_img_data= {
'final-base-delshot+rest-GE15':["{}/combine_add_base-shot+Asia-b3+beid+cap+mul+Chinap2+Pad2+Padb3+XCH_GE15_list_label.txt".format(final_mtcnn_root_path), 41233],
}
