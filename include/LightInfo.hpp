#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

enum LightColor
{
    RED,
    GREEN
};

enum ArrowType
{
    CIRCLE,
    LEFT,
    RIGHT,
    STRAIGHT
};

struct LightInfo
{
    Rect box;
    LightColor color;
    ArrowType direction;
};
