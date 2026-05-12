#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <deque>

#include "LightInfo.hpp"

using namespace std;
using namespace cv;

class PreProcess
{
public:
    PreProcess();

    vector<LightInfo> process(const Mat& frame);

    vector<vector<Point>> getRedContours() const { return red_contours; }
    vector<vector<Point>> getGreenContours() const { return green_contours; }

private:
    void loadTemplates();

    Mat createMask(const Mat& hsv, const Scalar& low, const Scalar& high);

    void cleanMask(Mat& mask);

    vector<Rect> findLightROIs(const Mat& mask);

    LightColor decideActiveColor(const vector<Rect>& red_rois,
                                  const vector<Rect>& green_rois);

    ArrowType matchArrow(const Mat& roi, LightColor color);

    Scalar low_red1, high_red1;
    Scalar low_red2, high_red2;
    Scalar low_green, high_green;
    double min_area, max_area;
    double match_threshold;
    int gaussian_k;
    int morph_k;
    Mat kernel;

    Mat red_mask, green_mask;
    vector<vector<Point>> red_contours, green_contours;

    map<LightColor, vector<pair<Mat, ArrowType>>> templates;

    deque<LightColor> history;
    int history_max;
    int vote_threshold;
};
