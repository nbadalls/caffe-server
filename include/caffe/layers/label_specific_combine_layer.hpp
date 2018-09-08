#ifndef CAFFE_LABEL_SPECIFIC_COMBINE_LAYER_HPP_
#define CAFFE_LABEL_SPECIFIC_COMBINE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

/**
 * @brief This layer adds a constant m within the cos() function of the Additive Angular Margin described in
 *         combine ArcFace with AMsoftmax face, remove the threshold condition
 *          @ https://arxiv.org/pdf/1801.07698.pdf.
 * @author lochappy : ttanloc@gmail.com
 *
 */

template <typename Dtype>
class LabelSpecificCombineLayer : public Layer<Dtype> {
 public:
  explicit LabelSpecificCombineLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelSpecificCombine"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype margin_m, margin_theta, sq_margin_theta, threshold, cos_margin_theta, sin_margin_theta;
};

}  // namespace caffe

#endif  // CAFFE_LABEL_COMBINE_LAYER_HPP_
