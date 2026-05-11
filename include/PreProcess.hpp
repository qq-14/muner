#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "LightInfo.hpp"

using namespace std;
using namespace cv;

class PreProcess{
public:
    PreProcess();
    
    // 定义识别函数
    vector<vector<Point>> detectLights(const Mat& frame);

    // 获取各颜色掩码
    Mat getRedMask() const { return red_mask;}
    Mat getGreenMask() const { return green_mask; }
    Mat getYellowMask() const { return yellow_mask; }

    // 各颜色轮廓获取
    vector<vector<Point>> getRedContours() const { return red_contours; }
    vector<vector<Point>> getGreenContours() const { return green_contours; }
    vector<vector<Point>> getYellowContours() const { return yellow_contours; }
    
private:
    Mat createMask(const Mat& hsv, const Scalar& low, const Scalar& high);

    void cleanMask(Mat& mask);

    vector<Rect> findLightROIs(const Mat& mask);

    LightColor decideActiveColor(const vector<Rect>& red_rois,
                                  const vector<Rect>& green_rois,
                                  const vector<Rect>& yellow_rois);

    ArrowType matchArrow(const Mat& roi, LightColor color);

    Scalar low_red1, high_red1;         // 红色区间
    Scalar low_red2, high_red2;         // 红色区间
    Scalar low_green, high_green;       // 绿色区间
    Scalar low_yellow, high_yellow;     // 黄色区间
    double min_area, max_area;          // 最大,最小面积
    double match_threshold;             // 识别阈值
    int gaussian_k;                     // 高斯模糊核大小
    int morph_k;                        // 高斯核大小
    Mat kernel;                         // 形态学核大小

    // 中间结果存储
    Mat red_mask, yellow_mask, green_mask; // 颜色掩码
    vector<vector<Point>> red_contours, yellow_contours, green_contours; // 颜色轮廓

    // 多帧稳定
    deque<LightColor> history;
    int history_max;
    int vote_threshold;
};