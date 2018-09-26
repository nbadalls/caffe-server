#include "caffe/extract_feature.hpp"

namespace caffe {


Extract_Feature::Extract_Feature(const std::string &param_proto_path)
{

    ReadProtoFromTextFileOrDie(param_proto_path, &param_);
    //init caffemodels and set means
    int model_num = param_.model_config_size();
    nets_.resize(model_num);
    means_.resize(model_num);
    scale_.resize(model_num);


    for(int i = 0; i < model_num; i++)
    {
        string deploy_path = param_.model_config(i).deploy_path();
        string model_path = param_.model_config(i).model_path();
        nets_[i].reset(new Net<float>(deploy_path, TEST));
        nets_[i]->CopyTrainedLayersFrom(model_path);

        //scale defalut as 1.0
        scale_[i] = param_.model_config(i).data_transform().scale();

        //set means
        if(param_.model_config(i).has_data_transform())
        {
            int mean_value_size = param_.model_config(i).data_transform().mean_value_size();
            if(mean_value_size >0)
            {
                for(int k = 0; k < mean_value_size; k++)
                {
                    float mean_val = param_.model_config(i).data_transform().mean_value(k);
                    means_[i].push_back(mean_val);
                }
            }
        }
    }

    //set run mode
    if(param_.has_run_mode())
    {
        if(param_.run_mode() == ExtractFeatureParameter_RunMode_CPU)
        {
            Caffe::set_mode(Caffe::CPU);
        }
        else if(param_.run_mode() == ExtractFeatureParameter_RunMode_GPU)
        {
            Caffe::set_mode(Caffe::GPU);
            if(param_.has_device_id())
            {
                Caffe::SetDevice(param_.device_id());
            }
        }
    }

    //Read image pair list
    string image_pair_path = param_.image_pair_list();
    bool ret  = ReadImagePairList(image_pair_path, image_pairs_);
    CHECK_EQ(ret, true) << "Read image pair file failed!!";

    //Read image list
    string image_list_path = param_.image_list();
    std::ifstream fin(image_list_path.c_str(), std::ios::in);
//    if(!fin.is_open()) return false;
    CHECK_EQ(fin.is_open(), true) << "Read image list file failed!!";
    string load_line;
    while(getline(fin, load_line))
    {
        vector<shared_ptr<Blob<float> > > features(model_num);
        image_features_.insert(std::make_pair(load_line, features));
    }
    fin.close();
}

void Extract_Feature::ImagePairExtractFeature()
{
    ExtractImageFeature();
    ComparePairs();
    SaveDist();
}

void Extract_Feature::ExtractImageFeature()
{
    int counter = 0;
    map<string, vector<shared_ptr<Blob<float> > > >::iterator iter_begin = image_features_.begin();
    while(iter_begin != image_features_.end())
    {
        string image_prefix = iter_begin->first;
        vector<shared_ptr<Blob<float> > > *feature_pt = &iter_begin->second;
        for(int i = 0; i < nets_.size(); i++)
        {
            CHECK_EQ(nets_[i]->num_inputs(), 1) << "number of input should be one";

            //load image
            string root_path = param_.model_config(i).image_root_path();
            string image_path = root_path + "/" + image_prefix;
            cv::Mat image = cv::imread(image_path);
            CHECK_EQ(image.empty(), false) << "image is empty: "<< image_path;
            //transform image into blob
            Blob<float>* input_blob_pt = nets_[i]->input_blobs()[0];
            ImageToBolb2(image, input_blob_pt, i);

            //output feature
            nets_[i]->Forward();

            //slience layer's input as feature --ps. slience has no output
            int output_index = nets_[i]->bottom_vecs().size()-1;
            Blob<float> *output_blob_pt = nets_[i]->bottom_vecs()[output_index][0];

            (*feature_pt)[i].reset(new Blob<float>(1, output_blob_pt->count(),1,1));

            //copy output blob into image_features_
            const float * src_pt = output_blob_pt->cpu_data();
            float * dst_pt = (*feature_pt)[i].get()->mutable_cpu_data();
            caffe_copy(output_blob_pt->count(), src_pt, dst_pt);


//            //for debug
//            std::cout << "[ ";
//            for(int k = 0; k < output_blob_pt->count(); k++)
//            {
//                std::cout <<*(dst_pt+k) << " ";
//            }
//            std::cout << " ]" << std::endl;
//            std::cout <<output_blob_pt->count() <<std::endl;
//            //end debug

        }

        std::cout << "Extract feature: " << counter << "/" << image_features_.size()-1 << "\r";
        iter_begin ++;
        counter++;
    }

}

void Extract_Feature::ComparePairs()
{
    int miss_count = 0;
    for(int i = 0; i < image_pairs_.size(); i++)
    {
        string lib_prefix = image_pairs_[i].lib_prefix;
        string cap_prefix = image_pairs_[i].cap_prefix;

        map<string, vector<shared_ptr<Blob<float> > > >::iterator  iter_lib = image_features_.find(lib_prefix);
        map<string, vector<shared_ptr<Blob<float> > > >::iterator  iter_cap = image_features_.find(cap_prefix);

        if(iter_lib != image_features_.end() && iter_cap != image_features_.end())
        {
            vector<shared_ptr<Blob<float> > > *lib_feature_pt = &(iter_lib->second);
            vector<shared_ptr<Blob<float> > > *cap_feature_pt = &(iter_cap->second);

            CHECK_EQ((*lib_feature_pt).size(), (*cap_feature_pt).size()) << "feature numbers mismatches..";

            //same pair with different models
            for(int j = 0; j < (*lib_feature_pt).size(); j++)
            {
                Blob<float>* lib_feature = (*lib_feature_pt)[j].get();
                Blob<float>* cap_feature = (*cap_feature_pt)[j].get();
                float square_dist = SquareEuclideanDist(lib_feature, cap_feature);

                //save Square Euclidean Dist
                image_pairs_[i].diffs.push_back(square_dist);
            }
        }
        else
        {
//            std::cout << "miss image feature.."<<std::endl;
            miss_count++;
        }

        std::cout << "Make comparation..: " << i << "/" << image_pairs_.size()-1 << "\r";
    }

    std::cout << std::endl<< "Sum " << miss_count << "image pair skiped.." << std::endl;
}

void Extract_Feature::SaveDist()
{

    int model_num = param_.model_config_size();
    vector<shared_ptr<std::ofstream> > fouts(model_num);

    //init out put stream
    for(int i = 0; i < model_num; i++)
    {
        string out_root_path = param_.model_config(i).output_path();
        string model_path = param_.model_config(i).model_path();
        string model_name = model_path.substr(model_path.find_last_of('/')+1);
        string model_prefix = model_name.substr(0, model_name.find(".caffemodel"));
//        string dist_result_path = out_root_path + "/result_v2_" + model_prefix + ".txt";
        char dist_result_path[1000];
        sprintf(dist_result_path, "%s/result_v2_%s.txt", out_root_path.c_str(), model_prefix.c_str());
        // std::cout << dist_result_path << std::endl;

//        fouts[i].open(dist_result_path, std::ios::out);
        fouts[i].reset(new std::ofstream(dist_result_path, std::ios::out));
        CHECK_EQ(fouts[i].get()->is_open(), true);
    }

    //save result into files
    for(int i = 0; i < image_pairs_.size(); i++)
    {
        for(int j = 0; j < image_pairs_[i].diffs.size(); j++)
        {
//            fouts[j] << image_pairs_[i].diffs[j] << " " << image_pairs_[i].label <<std::endl;
            (*fouts[j]) << image_pairs_[i].diffs[j] << " " << image_pairs_[i].label <<std::endl;
        }
    }

    for(int i = 0; i < model_num; i++)
    {
        (*fouts[i]).close();
    }
}

bool Extract_Feature::ReadImagePairList(const std::string &image_list_path,
                                               vector<caffe::Result> &image_pairs)
{

    std::ifstream fin(image_list_path.c_str(), std::ios::in);
    if(!fin.is_open()) return false;

    string load_line;
    while(getline(fin, load_line))
    {
        std::stringstream iss(load_line);
        Result  elem_result;
        iss >> elem_result.cap_prefix >> elem_result.lib_prefix >> elem_result.label;
        image_pairs.push_back(elem_result);
    }
    fin.close();
    return true;
}

void Extract_Feature::ImageToBolb(const cv::Mat &image, Blob<float> *image_blob)
{
    int height = image_blob->height();
    int width = image_blob->width();
    int channel = image_blob->channels();

    CHECK_EQ(height, image.rows) << "Height of bolb mismatches the rows of image";
    CHECK_EQ(width, image.cols) << "Width of bolb mismatches the cols of image";
    CHECK_EQ(channel, 3) << "channels of bolb should be three";

    float *pt_c1 = image_blob->mutable_cpu_data();
    float *pt_c2 = pt_c1 + height * width;
    float *pt_c3 = pt_c2 + height * width;
    cv::MatConstIterator_<cv::Vec3b> iter = image.begin<cv::Vec3b>();

    int index = 0;
    while(iter != image.end<cv::Vec3b>())
    {
        pt_c1[index] = static_cast<float>((*iter)[0]);
        pt_c2[index] = static_cast<float>((*iter)[1]);
        pt_c3[index] = static_cast<float>((*iter)[2]);
        index++;
        iter++;
    }
}

void Extract_Feature::ImageToBolb2(const cv::Mat &image, Blob<float> *image_blob)
{
    int height = image_blob->height();
    int width = image_blob->width();
    int channel = image_blob->channels();

    CHECK_EQ(height, image.rows) << "Height of bolb mismatches the rows of image";
    CHECK_EQ(width, image.cols) << "Width of bolb mismatches the cols of image";
    CHECK_EQ(channel, image.channels()) << "channels of bolb mismatches the channels of image";

    float *pt = image_blob->mutable_cpu_data();
    uchar *image_pt = image.data;
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            int index = h * width + w;
            for(int c = 0; c < channel; c++)
            {
                pt[index] = static_cast<float>(*image_pt);
                index += height* width;
                image_pt++;
            }
        }
    }
}

void Extract_Feature::ImageToBolb2(const cv::Mat &image, Blob<float> *image_blob, int net_id)
{
    int height = image_blob->height();
    int width = image_blob->width();
    int channel = image_blob->channels();

    CHECK_EQ(height, image.rows) << "Height of bolb mismatches the rows of image";
    CHECK_EQ(width, image.cols) << "Width of bolb mismatches the cols of image";
    CHECK_EQ(channel, image.channels()) << "channels of bolb mismatches the channels of image";

    float *pt = image_blob->mutable_cpu_data();
    uchar *image_pt = image.data;
    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            //bgr rank
            int index = h * width + w;
            for(int c = 0; c < channel; c++)
            {
                if(means_[net_id].size() >0 )
                {
                    CHECK_EQ(means_[net_id].size(), channel) << "means channel should equal with blob channels";
                    pt[index] = (static_cast<float>(*image_pt) - means_[net_id][c]) * scale_[net_id];
                }
                else
                {
                     pt[index] = static_cast<float>(*image_pt);
                }

                index += height* width;
                image_pt++;
            }
        }
    }
}



float Extract_Feature::SquareEuclideanDist( const Blob<float>* fea1,  const Blob<float>* fea2)
{
    CHECK_EQ(fea1->count(), fea2->count()) << "Dim of feature is different";

    //allocate enough space to make copy work
    Blob<float> sub_feature(1,fea1->count(),1,1);
    caffe_sub(fea1->count(), fea1->cpu_data(), fea2->cpu_data(), sub_feature.mutable_cpu_data());

    float dist = caffe_cpu_dot(fea1->count(), sub_feature.mutable_cpu_data(), sub_feature.mutable_cpu_data());
    return dist;
}

void Extract_Feature::ShowFeature(Blob<float> *blob_result)
{
    std::cout << "Feature: ";
    float *data = blob_result->mutable_cpu_data();
    for(int i = 0; i < blob_result->count(); i++)
    {
        std::cout << *data << " ";
        data++;
    }

    std::cout << std::endl;
}

}

