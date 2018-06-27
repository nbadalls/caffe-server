#define training set 

#b(base) -- shot_Oriental_Age_Lan_DHUA_PAKJ_Indon_Migrant_celeb
#FaceAdd -- ZheD-_Cmul-Asia-beid-cap10
#MS -- MSceleb 

image_root_folder = "/home/zkx/Data_sdb/TrainData/Data_sdc/Patches/MultiPatches"
label_root_path = "/home/zkx/Data_sdb/TrainData/Data_sdc/Patches/MultiPatches/image_list/train_list"
train_img_data = {

        'b+FaceAdd+Mobile': ["{}/Combine_base_Cmul-Asia-beid-cap10-ZheD-CHAMob_list.txt_label.txt".format(label_root_path),55918],
	'b+FaceAdd': [ "{}/Combine_base_Cmul-Asia-beid-cap10-ZheD_list.txt_label.txt".format(label_root_path), 54547], 
	'base' :['{}/shot_Oriental_Age_Lan_DHUA_PAKJ_Indon_Migrant_celeb_label.txt'.format(label_root_path), 21331],
}

lmdb_root_path =  "/home/zkx/Data_sdb/TrainData/Data_sdc/Patches/lmdb"
train_lmdb_data = {
	'b+FaceAdd+encode': [ "{}/fc_0.35_112x96_base_ZheD_Cmul-beid_cap10_Asia-train-encode-shufflelmdb".format(lmdb_root_path), 54547],
}


image_mtcnn_root_folder = "/mnt/glusterfs/o2n/s1/Data/Train_Data/O2N/Data_sdd/Patches_mtcnn/MultiPatches"
label_mtcnn_root_path = "/mnt/glusterfs/o2n/s1/Data/Train_Data/O2N/Data_sdd/Patches_mtcnn/MultiPatches_list/combine_folder_list"

train_mtcnn_img_data = {

        'b+add1+2': ["{}/combine_mtcnn_base+add1-2_GE8_list_label.txt".format(label_mtcnn_root_path), 67355],
}

