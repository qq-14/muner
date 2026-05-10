#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

enum LightColor
{
    RED,
    YELLO,
    GREEN
};

enum ArrowType
{
    CIRCLE,     // 圆形信号灯
    LEFT,       // 左转箭头
    RIGHT,      // 右转箭头
    STRAIGHT,   // 直行箭头
    UTURN       // 掉头箭头
};

struct  
{
    Rect box;
    LightColor color;
    ArrowType direction;
};
