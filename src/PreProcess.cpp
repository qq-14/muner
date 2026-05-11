#include "PreProcess.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

PreProcess::PreProcess() : gaussian_k(5), morph_k(5), min_area(30), max_area(5000), match_threshold(0.6),
                            history_max(10), vote_threshold(7)
{
    // 红色 HSV 区间
    low_red1  = Scalar(0,  120, 70);
    high_red1 = Scalar(10, 255, 255);
    low_red2  = Scalar(170, 120, 70);
    high_red2 = Scalar(180, 255, 255);

    // 绿色 HSV 区间
    low_green  = Scalar(35, 50, 50);
    high_green = Scalar(90, 255, 255);

    // 黄色 HSV 区间
    low_yellow  = Scalar(15, 100, 100);
    high_yellow = Scalar(35, 255, 255);

    // 形态学核
    kernel = getStructuringElement(MORPH_RECT, Size(morph_k, morph_k));
}