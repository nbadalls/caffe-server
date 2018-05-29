#use to create face extractor param
from __future__ import print_function
import sys
sys.path.append('/home/minivision/SoftWare/caffe-server/python')
import caffe_pb_feature



def create_param(device_id_, image_list_, image_pair_list_, model_infos_):

    model_param_config = []

    for elem  in model_infos_:

        if(elem.has_key('mean_value')):
            each_param = caffe_pb_feature.ModelInitParameter(
                    image_root_path = elem['image_root_path'],
                    deploy_path = elem['deploy_path'],
                    model_path = elem['model_path'],
                    output_path = elem['output_path'],
                    data_transform = caffe_pb_feature.TransformationParameter(
                        mean_value = elem['mean_value'],
                        scale = elem['scale']
                        )
                    )
        else:

            each_param = caffe_pb_feature.ModelInitParameter(
                    image_root_path = elem['image_root_path'],
                    deploy_path = elem['deploy_path'],
                    model_path = elem['model_path'],
                    output_path = elem['output_path']
                    )
        model_param_config.append(each_param)


    feature = caffe_pb_feature.ExtractFeatureParameter(
        run_mode = caffe_pb_feature.ExtractFeatureParameter.RunMode.Value('GPU'),
        device_id = device_id_,
        image_list = image_list_,
        image_pair_list = image_pair_list_,
        model_config = model_param_config
    )
    return feature

if __name__ == '__main__':

    device_id_ = 0
    image_list_ = "/media/minivision/OliverSSD/FaceRecognition/TestSet/XCH_PAD_01_23/image_list.txt"
    image_pair_list_ = "/media/minivision/OliverSSD/FaceRecognition/TestSet/XCH_PAD_01_23/image_pair_list.txt"
    # model_infos_ = [
    #     {
    #     'image_root_path':"/media/minivision/OliverSSD/FaceRecognition/TestSet/XCH_PAD_01_23/patches/XCH-Ad/fc_0.35_112x96",
    #     'deploy_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-28-2/MobileFaceNet_deploy.prototxt",
    #     'model_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-28-2/MobileFaceNet.caffemodel",
    #     'output_path':"/home/minivision/SoftWare/caffe-server/scripts/face_feature_extractor_result",
    #     'mean_value':[127.5, 127.5, 127.5],
    #     'scale': 0.0078125
    #     }
    # ]


    model_infos_ = [
        {
        'image_root_path':"/media/minivision/OliverSSD/FaceRecognition/TestSet/XCH_PAD_01_23/patches/XCH-Ad/fc_0.35_112x96",
        'deploy_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-29/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/mobileNet-bn_deploy.prototxt",
        'model_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-29/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/2018-05-24_AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn_zkx_iter_65000.caffemodel",
        'output_path':"/home/minivision/SoftWare/caffe-server/scripts/face_feature_extractor_result",
        # 'mean_value':[127.5, 127.5, 127.5],
        # 'scale': 0.0078125
        },

        # {
        # 'image_root_path':"/media/minivision/OliverSSD/FaceRecognition/TestSet/XCH_PAD_01_23/patches/XCH-Ad/fc_0.35_112x96",
        # 'deploy_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-29/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/mobileNet-bn_deploy.prototxt",
        # 'model_path':"/media/minivision/OliverSSD/FaceRecognition/verification_select_best_models/2018-05-29/AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet/2018-05-24_AMImageCdata-b0.3s30_fc_0.35_112x96_b+FaceAdd_MobileFaceNet-bn_zkx_iter_60000.caffemodel",
        # 'output_path':"/home/minivision/SoftWare/caffe-server/scripts/face_feature_extractor_result",
        # # 'mean_value':[127.5, 127.5, 127.5],
        # # 'scale': 0.0078125
        # }
    ]
    feature_config = create_param(device_id_, image_list_, image_pair_list_, model_infos_)
    dst_path = '/home/minivision/SoftWare/caffe-server/scripts/face_feature_extractor/param.prototxt'
    f = open(dst_path, 'w')
    print(feature_config, file = f)
    f.close()
