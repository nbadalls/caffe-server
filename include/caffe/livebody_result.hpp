/*
 *Livebody test with 4 classes
 * 0 fake face from paper
 * 1 true face
 * 2 fake face from ID
 * 3 fake face from screen
 */

#include "caffe/softmax_test.hpp"


namespace caffe {

class Livebody_Result : public SoftMax_Test
{
public:
    Livebody_Result(const string & param_path);
    void SaveTrueFaceResult();

private:

};

}
