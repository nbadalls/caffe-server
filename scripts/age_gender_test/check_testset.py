#use to check test set clean or not

import numpy as np
import os
import shutil
class CheckTestSet():
    def __init__(self, image_root_path, image_list, predict_result, dst_path):
        self.image_root_path = image_root_path
        self.image_list = image_list
        self.predict_result = predict_result
        self.dst_path = dst_path


    def check_test_set(self):
        f = open(self.image_list, 'r')
        image_list = f.read().splitlines()
        f.close()

        f = open(self.predict_result, 'r')
        data = f.read().splitlines()
        f.close()

        data_array = []
        for elem in data:
            parts = elem.split(' ')
            data_array.append([[float(part)] for part in parts])

        data_array = np.array(data_array)
        gender_label = data_array[:, -1]
        gender_predict_confidence = data_array[:, -4:-2]
        gender_predict_result = np.argmax(gender_predict_confidence, 1)
        compare_result = gender_predict_result == gender_label
        for index, select_elem in enumerate(compare_result):
            if not select_elem:
                print(image_list[index].split(' ')[0])
                image_path = '{}/{}'.format(image_root_path, image_list[index].split(' ')[0])
                gender_prefix = image_list[index].split(' ')[0].split('/')[-2]
                basename = image_list[index].split(' ')[0].split('/')[-1]
                dst_image_path = '{}/{}/{}'.format(dst_path, gender_prefix, basename)
                dst_parent_path = '{}/{}'.format(dst_path, gender_prefix)
                if not os.path.exists(dst_parent_path):
                    os.makedirs(dst_parent_path)
                shutil.copy(image_path, dst_image_path)


if __name__ == '__main__':
    image_root_path = "/media/minivision/OliverSSD/GenderAge/Test/TestSet-patches/fc_0.25_90x80"
    image_list = "/media/minivision/OliverSSD/GenderAge/Test/TestSet-patches/image_list/combine_list/combine_Test1-4+2017+2018-08-09_testset_softmax_label.txt"
    predict_result = "/media/minivision/OliverSSD/GenderAge/best_models/2018-08-17/Result_age-gender1-b/result_age-3.6443-gender-0.9342_2018-08-17_AgeGenderMtcnn_fc_0.25_112x96_age-gender-dataset1_MobileFaceNet_zkx_iter_130000.txt"
    dst_path = "/media/minivision/OliverSSD/GenderAge/Test/TestSetCheck/b_2"
    check_test = CheckTestSet(image_root_path, image_list, predict_result, dst_path)
    check_test.check_test_set()
