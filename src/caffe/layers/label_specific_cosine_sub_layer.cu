#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/label_specific_cosine_sub_layer.hpp"

namespace caffe {


  template <typename Dtype>
  __global__ void LabelSpecificCosineSubForward(const int n, const int dim, const Dtype* label,
                                                const Dtype* bottom_data, Dtype* top_data,
                                                const Dtype cos_margin_theta, const Dtype sin_margin_theta,
                                                const Dtype sq_margin_theta , const Dtype margin_m ,
                                                const Dtype threshold) {
    CUDA_KERNEL_LOOP(i, n) {
      const int gt = static_cast<int>(label[i]);
      const int idx = i * dim + gt;
      const Dtype cos_theta = bottom_data[idx];
      Dtype tmp = cos_theta;
      if (cos_theta > threshold)
      {
           const Dtype sq_cos_theta = cos_theta*cos_theta;
           if ((sq_cos_theta <= (Dtype)1.) && (sq_cos_theta >= (Dtype)-1.)){
               const Dtype sin_theta = sqrt((Dtype)1. - sq_cos_theta);
               tmp = cos_theta*cos_margin_theta - sin_theta*sin_margin_theta;  //modified by zkx 2018-03-23
           }
      } else {
          tmp = cos_theta - sq_margin_theta;
      }

      if (tmp > (-margin_m))
          tmp += margin_m;

      top_data[idx] = tmp;
    }
  }

  template <typename Dtype>
  void LabelSpecificCosineSubLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int num = bottom[0]->num();
    const int count = bottom[0]->count();
    const int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
    if (this->phase_ == TEST) return;

    if (top.size() == 2) {
      top[1]->mutable_cpu_data()[0] = margin_theta;
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificCosineSubForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, bottom_data, top_data, cos_margin_theta, sin_margin_theta,sq_margin_theta,margin_m,threshold);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void LabelSpecificCosineSubBackward(const int n, const int dim, const Dtype* label,
                                                const Dtype* bottom_data, const Dtype* top_diff,
                                                Dtype* bottom_diff, const Dtype cos_margin_theta,
                                                const Dtype sin_margin_theta, const Dtype threshold) {
    CUDA_KERNEL_LOOP(i, n) {
      const int gt = static_cast<int>(label[i]);
      const int idx = i * dim + gt;
      const Dtype cos_theta = bottom_data[idx];
      //clip cos_theta =[-1,1]
      if (cos_theta > threshold)
      {
          const Dtype sq_cos_theta = cos_theta*cos_theta;
          if ((sq_cos_theta <= (Dtype)1.) && (sq_cos_theta >= (Dtype)-1.)){
              const Dtype sin_theta = sqrt((Dtype)1. - sq_cos_theta) + (Dtype)1e-6;
              bottom_diff[idx] = (cos_margin_theta + sin_margin_theta*(cos_theta/sin_theta))*top_diff[idx]; //modified by zkx 2018-03-23
          }
      }

    }
  }

  template <typename Dtype>
  void LabelSpecificCosineSubLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (top[0] != bottom[0] && propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const int count = bottom[0]->count();
      caffe_copy(count, top_diff, bottom_diff);

      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* label_data = bottom[1]->gpu_data();
      const int num = bottom[0]->num();
      const int dim = count / num;
      // NOLINT_NEXT_LINE(whitespace/operators)
      LabelSpecificCosineSubBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, label_data, bottom_data, top_diff, bottom_diff, cos_margin_theta, sin_margin_theta, threshold);
      CUDA_POST_KERNEL_CHECK;
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificCosineSubLayer);
}  // namespace caffe
