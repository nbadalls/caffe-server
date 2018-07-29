#include "caffe/softmax_test.hpp"


namespace caffe {

 SoftMax_Test::SoftMax_Test(const std::string &param_proto_path)
{

    ReadProtoFromTextFileOrDie(param_proto_path, &param_);

    //init net
    batch_size_ = param_.batch_size();
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
        CHECK_EQ(batch_size_, nets_[i]->input_blobs()[0]->num()) << "batch size should be equal to input number";
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

 void SoftMax_Test::ImageToBolb_batch(const cv::Mat &image, int offset, int net_id)
 {
     Blob<float> * input_bolb_pt = nets_[net_id]->input_blobs()[0];
     int height = input_bolb_pt->height();
     int width = input_bolb_pt->width();
     int channel = input_bolb_pt->channels();

     CHECK_EQ(height, image.rows) << "Height of bolb mismatches the rows of image";
     CHECK_EQ(width, image.cols) << "Width of bolb mismatches the cols of image";
     CHECK_EQ(channel, image.channels()) << "channels of bolb mismatches the channels of image";

     float *pt = input_bolb_pt->mutable_cpu_data()+offset;
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

 void SoftMax_Test::Predict_batch()
 {
    int batch_num = ceil(float(image_label_list_.size()) / float(batch_size_));
    std::cout << "batch size is : " << batch_size_ << std::endl;

    //init presult
    presult_.resize(batch_num*batch_size_);
    for(int bnum = 0; bnum < batch_num; bnum++)
    {
        for(int net_id = 0; net_id < nets_.size(); net_id++)
        {
            CHECK_EQ(nets_[net_id]->num_inputs(), 1) << "number of input should be one";
            string image_root_path = param_.model_config(net_id).image_root_path();
            Blob<float> * image_bolb_pt = nets_[net_id]->input_blobs()[0];
            for(int item_id = 0; item_id < batch_size_; item_id++)
            {
                int image_id = bnum * batch_size_ + item_id;

                //number parts beyond the image list get last image repeated
                if(image_id > image_label_list_.size()-1)
                {
                    image_id =image_label_list_.size()-1;
                }

                cv::Mat image = cv::imread(image_root_path + "/" + image_label_list_[image_id].first);
                CHECK_EQ(image.empty(), false) << "image is empty: "<< image_root_path << std::endl << image_label_list_[image_id].first;
                int offset =  image_bolb_pt->offset(item_id);
                ImageToBolb_batch(image,  offset,  net_id);

                if(image_id %100 == 0)
                {
                    std::cout << "Complete-image.. " << image_id << "/"
                              <<image_label_list_.size()-1
                              << "   [" << int(float(image_id*100)/float(image_label_list_.size()-1))
                              << "%]" << std::flush;
                }

            }

            //output predict result
            nets_[net_id]->Forward();
            //last softmax layer's result as output
            Blob<float> * output_blob_pt = nets_[net_id]->output_blobs()[0];
            int result_offset = output_blob_pt->count() / batch_size_;
            for(int item_id = 0; item_id < batch_size_; item_id++)
            {
                float* begin = output_blob_pt->mutable_cpu_data() + item_id * result_offset;
                float* end = begin + result_offset;
                vector<float> result_item(begin, end);

                vector<float> origin_predict_item(4);
                //get origin label predict result
                int image_id = bnum * batch_size_ + item_id;
                //number parts beyond the image list get last image repeated
                if(image_id > image_label_list_.size()-1)
                {
                    image_id =image_label_list_.size()-1;
                }

                int label = image_label_list_[image_id].second;
                float origin_result = result_item[label];

                origin_predict_item[0] = label;
                origin_predict_item[1] = origin_result;

                //get best prefict label
                int prefict_label = 0;
                float prefict_possibility = result_item[0];

                for(int i = 0; i < result_item.size(); i++)
                {
                    if (result_item[i] > prefict_possibility)
                    {
                        prefict_possibility = result_item[i];
                        prefict_label = i;
                    }
                }

                origin_predict_item[2] = prefict_label;
                origin_predict_item[3] = prefict_possibility;

                //save origin label and possibility predict label and possibility
                int result_id = bnum * batch_size_ + item_id;
                if (presult_[result_id].size() == 0)
                {
                    presult_[result_id].resize(nets_.size());
                }
                presult_[result_id][net_id].push_back((origin_predict_item));
            }
        }

//        std::cout << "Complete.. " << bnum << "/" << batch_num-1 << "\r" << std::flush;
    }


 }


}
