#include "caffe/softmax_test.hpp"


namespace caffe {

 SoftMax_Test::SoftMax_Test(const std::string &param_proto_path)
{

    ReadProtoFromTextFileOrDie(param_proto_path, &param_);

    //init net
    int num_model = param_.model_config_size();
    scale_.resize(num_model);
    means_.resize(num_model);
    nets_.resize(num_model);

    for(int i = 0; i < num_model; i++)
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

    //get image list
    string image_list_path =param_.image_list();
    std::ifstream fin(image_list_path.c_str(), std::ios::in);
    CHECK_EQ(fin.is_open(), true) << "Open image list failed!! " << image_list_path;
    string line;
    while(std::getline(fin, line))
    {
        stringstream iss(line);
        string image_path;
        int label;
        iss >> image_path >> label;
        image_label_list_.push_back(make_pair(image_path, label));
    }
    fin.close();
 }

 void SoftMax_Test::ImageToBolb2(const cv::Mat &image, Blob<float> *image_blob, int net_id)
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
             int index = h * width + w ;
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

 void SoftMax_Test::Predict()
 {
     for(int i = 0; i < image_label_list_.size(); i++)
     {
         vector<SoftmaxResult> nets_result;
        for(int j = 0; j < nets_.size(); j++)
        {
            CHECK_EQ(nets_[j]->num_inputs(), 1) << "number of input should be one";

            string image_root_path = param_.model_config(j).image_root_path();
            cv::Mat image = cv::imread(image_root_path + "/" + image_label_list_[i].first);
            CHECK_EQ(image.empty(), false) << "image is empty: "<< image_root_path << std::endl << image_label_list_[i].first;
            Blob<float> * image_bolb_pt = nets_[j]->input_blobs()[0];

            ImageToBolb2(image, image_bolb_pt, j);

            //output predict result
            nets_[j]->Forward();

            //last softmax layer's result as output
            Blob<float> * output_blob_pt = nets_[j]->output_blobs()[0];

            //save prefict result
            SoftmaxResult temp_result(1);
            int count = output_blob_pt->count();
            float * begin = output_blob_pt->mutable_cpu_data();
            float * end = begin + count;
            temp_result[0] = vector<float>(begin, end);

//            for(int k = 0; k < output_blob_pt->count(); k++)
//            {
//                float prefict_elem = *(output_blob_pt->cpu_data() + k);
//                temp_result[0].push_back(prefict_elem);
//            }
            nets_result.push_back(temp_result);
        }
         presult_.push_back(nets_result);

        std::cout << "Complete.. " << i << "/" << image_label_list_.size()-1 << "\r" << std::flush;
     }
 }


}