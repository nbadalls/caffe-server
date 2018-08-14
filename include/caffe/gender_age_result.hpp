/*
* gender age test with multi-label
* age 0-15 16-25 26-35 36-45 46-55 56-65
*
*
 */

#include "caffe/softmax_test.hpp"


namespace caffe {

class Age_Gender_Result : public SoftMax_Test
{
public:
    Age_Gender_Result(const string & param_path);
    void SaveTrueFaceResult();

private:
    //first  item is average bias of age
    //second  item is accuracy of gender
      std::vector<vector<float> > get_age_gender_accuracy();
      int age_transform(const std::vector<float> & age_presult);
//      void unify_result(std::vector<float> & presult);

};

}
