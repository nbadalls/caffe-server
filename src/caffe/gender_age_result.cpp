#include "caffe/gender_age_result.hpp"
#include <numeric>

namespace caffe {

Age_Gender_Result::Age_Gender_Result(const std::string &param_path):SoftMax_Test(param_path)
{}

void Age_Gender_Result::SaveTrueFaceResult()
{
    vector<vector<SoftmaxResult> > presult = get_presult();
    ExtractFeatureParameter param = get_param();

    int model_num = param.model_config_size();
    vector<shared_ptr<std::ofstream> > fout_v(model_num);

    //get model accuracy
    std::vector<vector<float> > prefict_accuracy = get_age_gender_accuracy();
    //init ofstream
    for(int i = 0; i < model_num; i++)
    {
        string output_path = param.model_config(i).output_path();
        string model_path = param.model_config(i).model_path();
        string model_name = model_path.substr(model_path.find_last_of('/')+1);
        string model_prefix = model_name.substr(0, model_name.find_last_of("."));
        float aver_age_bias = prefict_accuracy[i][0];
        float gender_accu = prefict_accuracy[i][1];

        char result_path[1000];
        sprintf(result_path, "%s/result_age-%.4f-gender-%.4f_%s.txt",
                output_path.c_str(),aver_age_bias,  gender_accu, model_prefix.c_str());

        fout_v[i].reset(new std::ofstream(result_path, ios::out));
        CHECK_EQ(fout_v[i]->is_open(), true) << "Open fail: " << result_path;
    }

    // models
    std::cout << "save result " << std::endl;
    vector<std::pair<string, std::vector<int> > > image_label_list = get_image_label_list();

    //image list
    for(int i = 0; i <  presult.size(); i++)
    {
        int age_label = image_label_list[i].second[0];
        int gender_label = image_label_list[i].second[1];

        //different model result
        for(int j = 0; j < presult[i].size(); j++)
        {
            vector<float> age_presult = presult[i][j][0];
            vector<float> gender_presult = presult[i][j][1];

            CHECK_EQ(age_presult.size(), 6) << "Items of age is 6";
            CHECK_EQ(gender_presult.size(), 2) << "Items of gender is 2";

            //save age gender predict result
            for(int age_indx = 0; age_indx < age_presult.size(); age_indx++)
            {
                (*fout_v[j]) << age_presult[age_indx] << " ";
            }

            (*fout_v[j]) << gender_presult[0] << " " << gender_presult[1] << " "
                         << age_label << " " <<gender_label << std::endl;
        }
    }

    //close
    for(int i = 0; i < model_num; i++)
    {
        fout_v[i]->close();
    }
}

std::vector<vector<float> > Age_Gender_Result::get_age_gender_accuracy()
{
     vector<vector<SoftmaxResult> > presult = get_presult();
     vector<std::pair<string, std::vector<int> > > image_label_list = get_image_label_list();

     int img_num =presult.size();
     int model_num = presult[0].size();

     //first  item is average bias of age
     //second  item is accuracy of gender
     std::vector<vector<float> > age_gender_accuracy_result(presult.size());
     for(int i = 0; i < model_num; i++)
     {
        int gender_true_predict = 0;
        float age_bias = 0;
        int age_num = 0;
        for(int j = 0; j < img_num; j++)
         {
             vector<float> age_result = presult[j][i][0];
             vector<float> gender_result = presult[j][i][1];
             int predict_age = age_transform(age_result);
             int predict_gender = gender_result[0] >= gender_result[1] ? 0 : 1;

             float label_age = image_label_list[j].second[0];
             int label_gender = image_label_list[j].second[1];

             if(predict_gender == label_gender)
             {
                 gender_true_predict+=1;
             }
             if((label_age + 1.0) > 1e-5)
             {
                 age_bias += abs(predict_age - label_age);
                 age_num++;
             }
         }
        float gender_accuracy = float(gender_true_predict) / float(img_num);
        float average_age_bias = float(age_bias) / float(age_num);

        age_gender_accuracy_result[i].push_back(average_age_bias);
        age_gender_accuracy_result[i].push_back(gender_accuracy);
     }
    return age_gender_accuracy_result;
}


//void Age_Gender_Result::unify_result(std::vector<float> &presult)
//{
//    float sum = std::accumulate(presult.begin(), presult.end(), 0);
//    for(int i = 0; i < presult.size(); i++)
//    {
//        presult[i] = presult[i]/sum;
//    }
//}

int Age_Gender_Result::age_transform(const std::vector<float> & age_presult)
{
    float predict_age = 0;
    for(int i = 0; i < age_presult.size(); i++)
    {
        if(i == 0)
        {
            predict_age += 7.5 * age_presult[i];
        }
        else
        {
            predict_age += (20.5 + 10*(i-1)) * age_presult[i];
        }
    }
    return int(predict_age);
}

}
