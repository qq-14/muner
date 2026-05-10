#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class PreProcess{
public:
    // 定义预处理函数
    Mat process(const Mat& frame);
    // 定义识别函数
    
private:
    int gaussian_k = 5; // 高斯模糊核大小
    int morph_k = 5;    // 高斯核大小
    // 红色区间1
    Scalar low_red1  = Scalar(0, 120, 70);
    Scalar high_red1 = Scalar(10, 255, 255);
    // 红色区间2
    Scalar low_red2  = Scalar(170,120,70);
    Scalar high_red2 = Scalar(180,255,255);
    // 绿色区间
    Scalar low_green  = Scalar(35,50,50);
    Scalar high_green = Scalar(90,255,255);
    // 黄色区间
    Scalar low_yellow  = Scalar(15,100,100);
    Scalar high_yellow = Scalar(35,255,255);
};