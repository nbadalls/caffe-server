//Use to extract face feature from image pairs
//Author Zkx@__@
//Date 2018-05-28


#include "caffe/extract_feature.hpp"

//using namespace caffe;

using namespace caffe;

int main(int argc ,char* argv[])
  {

   FLAGS_alsologtostderr = 0;  // Print output to stderr (while still logging)
   ::google::InitGoogleLogging(argv[0]);
   if(argc < 2)
   {
       std::cout << "Please input Configuration file path"<<std::endl;
       return -1;
   }

   string config_path = argv[1];
    // string config_path = "/home/minivision/SoftWare/caffe-server/scripts/face_feature_extractor/param.prototxt";
    Extract_Feature feature_extractor(config_path);
    feature_extractor.ImagePairExtractFeature();
    return 0;
 }


