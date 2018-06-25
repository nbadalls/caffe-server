/*
 * Use to predict result of livebody result
 * Author Zkx@__@
 * Data 2018-06-24
 */


#include "caffe/livebody_result.hpp"


using namespace caffe;


int main(int argc, char*argv[])
{
    FLAGS_alsologtostderr = 0;  // Print output to stderr (while still logging)
    ::google::InitGoogleLogging(argv[0]);
    if(argc < 2)
    {
        std::cout << "Please input Configuration file path"<<std::endl;
        return -1;
    }

    Livebody_Result livebody_predictor(argv[1]);
//    Livebody_Result livebody_predictor("/home/minivision/SoftWare/caffe-server/tools/test_config_file_E1.prototxt");
    livebody_predictor.Predict();
    livebody_predictor.SaveTrueFaceResult();

    return 0;
}
