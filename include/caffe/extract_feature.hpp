/*
* Use to extract feature on the face recognition
* Date 2018-05-25
* Author Zkx@__@
*/
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caffe {

typedef struct Result_
{
    // create image reconition pairs
    // cap_prefix lib_prefix label ..
    // could read from list file
    string lib_prefix;
    string cap_prefix;
    string label;

    //feature compares scores
    //more than one model to test
    vector<float> diffs;

} Result;


class Extract_Feature
{
public:
  Extract_Feature(const string & param_proto_path);
  ~Extract_Feature(){}

  void ImagePairExtractFeature();

  void ExtractImageFeature();
  void ComparePairs();
  void SaveDist();
  //use to read image pair list from file
  // image1 image2 label ...
  bool ReadImagePairList(const string & image_list_path, vector<Result> &image_pairs);

  void ImageToBolb(const cv::Mat &image, Blob<float> * image_blob);
  void ImageToBolb2(const cv::Mat &image, Blob<float> * image_blob);
  //subtract means if it exists..
  void ImageToBolb2(const cv::Mat &image, Blob<float> * image_blob, int net_id);

  //l2 distance without sqrt
  float SquareEuclideanDist( const Blob<float>* fea1,  const Blob<float>* fea2); //const data

  //debug tools
  void ShowFeature(Blob<float>* blob_result);

private:
    vector<Result> image_pairs_;
    vector<shared_ptr<Net<float> > > nets_;
    map<string, vector<shared_ptr<Blob<float> > > > image_features_;
    ExtractFeatureParameter param_;
    vector<vector<float> > means_; //matches with multi-nets
    vector<float> scale_;
};
}

