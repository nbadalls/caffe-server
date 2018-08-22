#include <algorithm>
#include <vector>

#include "caffe/layers/label_specific_cosine_sub_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelSpecificCosineSubLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const LabelSpecificCosineSubParameter& param = this->layer_param_.label_specific_cosine_sub_param();
    if (param.has_margin_theta())
        margin_theta = param.margin_theta();
    else
        margin_theta = (Dtype)0.5;
    cos_margin_theta = cos(margin_theta);
    sin_margin_theta = sin(margin_theta);
    sq_margin_theta = sin(M_PI-margin_theta)*margin_theta;
    threshold = cos(M_PI-margin_theta);

    if (param.has_margin_m())
        margin_m = param.margin_m();
    else
        margin_m = (Dtype)-0.35;
}

template <typename Dtype>
void LabelSpecificCosineSubLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {
    if(top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
    vector<int> shape(1,1);
    if (top.size() == 2) top[1]->Reshape(shape);
}

template <typename Dtype>
void LabelSpecificCosineSubLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const int num = bottom[0]->num();
    const int count = bottom[0]->count();
    const int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);

    if (this->phase_ == TEST) return;

    if (top.size() == 2) {
        top[1]->mutable_cpu_data()[0] = margin_theta;
    }

    for (int i = 0; i < num; ++i) {
        const int gt = static_cast<int>(label_data[i]);
        const int idx = i * dim + gt;
        const Dtype cos_theta = bottom_data[idx];
        Dtype tmp = cos_theta;
        if (cos_theta > threshold)
        {
            const Dtype sq_cos_theta = cos_theta*cos_theta;
            if ((sq_cos_theta <= (Dtype)1.) && (sq_cos_theta >= (Dtype)-1.)){
                const Dtype sin_theta = sqrt((Dtype)1. - sq_cos_theta);
                tmp = cos_theta*cos_margin_theta - sin_theta*sin_margin_theta; //modified by zkx 2018-03-23
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
void LabelSpecificCosineSubLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom) {
    if (top[0] != bottom[0] && propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        caffe_copy(count, top_diff, bottom_diff);

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        const int num = bottom[0]->num();
        const int dim = count / num;
        for (int i = 0; i < num; ++i) {
            const int gt = static_cast<int>(label_data[i]);
            const int idx = i * dim + gt;
            const Dtype cos_theta = bottom_data[idx];
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
}


#ifdef CPU_ONLY
STUB_GPU(LabelSpecificCosineSubLayer);
#endif

INSTANTIATE_CLASS(LabelSpecificCosineSubLayer);
REGISTER_LAYER_CLASS(LabelSpecificCosineSub);

}  // namespace caffe
