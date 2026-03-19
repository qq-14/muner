#pragma once

#include <opencv2/opencv,hpp>
#include <vector>

using namespace std;
using namespace cv;

class PreProcess{
public:
    Mat process(const Mat& frame);
    
}