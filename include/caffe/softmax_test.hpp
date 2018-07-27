/*
* Use to get softmax result
* Date 2018-06-21
* Author Zkx@__@
*/
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV


namespace caffe {

typedef vector<vector<float> > SoftmaxResult;

class SoftMax_Test{
    public:
    SoftMax_Test(const string & param_proto_path);
    void ImageToBolb2(const cv::Mat &image, Blob<float> * image_blob, int net_id);
    void ImageToBolb_batch(const cv::Mat &image, int offset, int net_id);

    void Predict();

    void Predict_batch();

    vector<vector<SoftmaxResult> > get_presult(){
        return presult_;
    }

    ExtractFeatureParameter get_param(){
        return param_;
    }

    vector<std::pair<string, int> > get_image_label_list(){
        return image_label_list_;
    }

private:
    ExtractFeatureParameter param_;
    vector<vector<float> > means_; //matches with multi-nets
    vector<float> scale_;
    vector<shared_ptr<Net<float> > > nets_;
    vector<vector<SoftmaxResult> > presult_;
    vector<std::pair<string, int> > image_label_list_;
    int batch_size_;
};

}

