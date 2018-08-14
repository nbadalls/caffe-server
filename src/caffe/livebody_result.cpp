#include "caffe/livebody_result.hpp"

namespace caffe {

Livebody_Result::Livebody_Result(const std::string &param_path):SoftMax_Test(param_path)
{}

void Livebody_Result::SaveTrueFaceResult()
{
    vector<vector<SoftmaxResult> > presult = get_presult();
    ExtractFeatureParameter param = get_param();

    int model_num = param.model_config_size();
    vector<shared_ptr<std::ofstream> > fout_v(model_num);

    //init ofstream
    for(int i = 0; i < model_num; i++)
    {
        string output_path = param.model_config(i).output_path();
        string model_path = param.model_config(i).model_path();
        string model_name = model_path.substr(model_path.find_last_of('/')+1);
        string model_prefix = model_name.substr(0, model_name.find_last_of("."));
        char result_path[1000];
        sprintf(result_path, "%s/result_livebody_%s.txt", output_path.c_str(), model_prefix.c_str());
        fout_v[i].reset(new std::ofstream(result_path, ios::out));
        CHECK_EQ(fout_v[i]->is_open(), true) << "Open fail: " << result_path;
    }

    // models
    std::cout << "save result " << std::endl;
    vector<std::pair<string, std::vector<int> > > image_label_list = get_image_label_list();
    for(int i = 0; i <  presult.size(); i++)
    {
        int label = image_label_list[i].second[0];
        //list
        for(int j = 0; j < presult[i].size(); j++)
        {
            float true_face_result = presult[i][j][0][1];
             (*fout_v[j]) << std::setprecision(6) << true_face_result<<" ";

//            (*fout_v[j]) << presult[i][j][0][0]<<" " << presult[i][j][0][1] << " "
//                         << presult[i][j][0][2] << " "<< presult[i][j][0][3] << " ";
            //possibility of true face
            if(label == 1)
            {
                (*fout_v[j]) << 1 << std::endl;
            }
            else
            {
                (*fout_v[j]) << 0 << std::endl;
            }
        }
    }

    //close
    for(int i = 0; i < model_num; i++)
    {
        fout_v[i]->close();
    }
}

void Livebody_Result::SaveTrueFaceResult2()
{
    vector<vector<SoftmaxResult> > presult = get_presult();
    ExtractFeatureParameter param = get_param();

    int model_num = param.model_config_size();
    vector<shared_ptr<std::ofstream> > fout_v(model_num);

    //init ofstream
    for(int i = 0; i < model_num; i++)
    {
        string output_path = param.model_config(i).output_path();
        string model_path = param.model_config(i).model_path();
        string model_name = model_path.substr(model_path.find_last_of('/')+1);
        string model_prefix = model_name.substr(0, model_name.find_last_of("."));
        char result_path[1000];
        sprintf(result_path, "%s/result_face_recognition_predict_%s.txt", output_path.c_str(), model_prefix.c_str());
        fout_v[i].reset(new std::ofstream(result_path, ios::out));
        CHECK_EQ(fout_v[i]->is_open(), true) << "Open fail: " << result_path;
    }

    // get models
    std::cout << "save result " << std::endl;
    vector<std::pair<string, std::vector<int> > > image_label_list = get_image_label_list();

    //save according to image list
    for(int image_id = 0; image_id <  image_label_list.size(); image_id++)
    {
        int label = image_label_list[image_id].second[0];
        string image_path = image_label_list[image_id].first;
        //list
        for(int j = 0; j < presult[image_id].size(); j++)
        {
            CHECK_EQ(label, int(presult[image_id][j][0][0]))
                    << "label in result list is different with label in image list";
            // save label's result
            float predict_label_result = presult[image_id][j][0][1];
               (*fout_v[j])<< image_path<< " " << label << " "
                            <<std::setprecision(6) << predict_label_result<<" ";

            //save predict's result
            int predict_label = int(presult[image_id][j][0][2]);
            float predict_possibility = presult[image_id][j][0][3];

            *fout_v[j] << predict_label << " " << predict_possibility << std::endl;
        }
    }

    //close
    for(int i = 0; i < model_num; i++)
    {
        fout_v[i]->close();
    }
}

}
