#include "PreProcess.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>

using namespace cv;
using namespace std;

PreProcess::PreProcess() : gaussian_k(3), morph_k(5), min_area(200), max_area(5000), match_threshold(0.6),
                            history_max(10), vote_threshold(7)
{
    low_red1  = Scalar(0,   120, 70);
    high_red1 = Scalar(10,  255, 255);
    low_red2  = Scalar(170, 120, 70);
    high_red2 = Scalar(180, 255, 255);

    low_green  = Scalar(35, 50, 50);
    high_green = Scalar(90, 255, 255);

    kernel = getStructuringElement(MORPH_RECT, Size(morph_k, morph_k));

    loadTemplates();
}

void PreProcess::loadTemplates()
{
    string base = TEMPLATE_DIR;
    vector<pair<LightColor, string>> color_map = {
        {RED,   base + "/red"},
        {GREEN, base + "/green"}
    };

    for (auto& [color, dir_path] : color_map)
    {
        if (!filesystem::exists(dir_path)) continue;
        for (auto& entry : filesystem::directory_iterator(dir_path))
        {
            string ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".png") continue;

            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            string name = entry.path().stem().string();
            transform(name.begin(), name.end(), name.begin(), ::tolower);

            ArrowType type;
            if (name.find("left") != string::npos)        type = LEFT;
            else if (name.find("right") != string::npos)   type = RIGHT;
            else if (name.find("straight") != string::npos) type = STRAIGHT;
            else continue;

            templates[color].push_back({img, type});
        }
    }
}

Mat PreProcess::createMask(const Mat& hsv, const Scalar& low, const Scalar& high)
{
    Mat mask;
    inRange(hsv, low, high, mask);
    return mask;
}

void PreProcess::cleanMask(Mat& mask)
{
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
}

vector<Rect> PreProcess::findLightROIs(const Mat& mask)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> rois;
    for (auto& c : contours)
    {
        double area = contourArea(c);
        if (area < min_area || area > max_area) continue;
        rois.push_back(boundingRect(c));
    }
    return rois;
}

LightColor PreProcess::decideActiveColor(const vector<Rect>& red_rois,
                                          const vector<Rect>& green_rois)
{
    double r = red_rois.size(), g = green_rois.size();
    if (r == 0 && g == 0) return RED;
    return (r >= g) ? RED : GREEN;
}

ArrowType PreProcess::matchArrow(const Mat& roi, LightColor color)
{
    auto it = templates.find(color);
    if (it == templates.end() || it->second.empty()) return CIRCLE;

    double best_val = -1.0;
    ArrowType best_type = CIRCLE;

    for (auto& [tmpl, type] : it->second)
    {
        Mat resized;
        resize(roi, resized, tmpl.size());

        Mat result;
        matchTemplate(resized, tmpl, result, TM_CCOEFF_NORMED);

        double minV, maxV;
        Point minP, maxP;
        minMaxLoc(result, &minV, &maxV, &minP, &maxP);

        if (maxV > best_val)
        {
            best_val = maxV;
            best_type = type;
        }
    }

    return (best_val >= match_threshold) ? best_type : CIRCLE;
}

vector<LightInfo> PreProcess::process(const Mat& frame)
{
    Mat blurred, hsv;
    GaussianBlur(frame, blurred, Size(gaussian_k, gaussian_k), 0);
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    Mat rm1 = createMask(hsv, low_red1, high_red1);
    Mat rm2 = createMask(hsv, low_red2, high_red2);
    red_mask = rm1 | rm2;
    green_mask = createMask(hsv, low_green, high_green);

    cleanMask(red_mask);
    cleanMask(green_mask);

    vector<vector<Point>> red_all, green_all;
    findContours(red_mask, red_all, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(green_mask, green_all, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    red_contours.clear(); green_contours.clear();
    for (auto& c : red_all)   if (contourArea(c) >= min_area && contourArea(c) <= max_area) red_contours.push_back(c);
    for (auto& c : green_all) if (contourArea(c) >= min_area && contourArea(c) <= max_area) green_contours.push_back(c);

    auto red_rois   = findLightROIs(red_mask);
    auto green_rois = findLightROIs(green_mask);

    LightColor active = decideActiveColor(red_rois, green_rois);

    history.push_back(active);
    if ((int)history.size() > history_max) history.pop_front();

    int votes[2] = {0};
    for (auto c : history) votes[c]++;

    int max_v = max(votes[0], votes[1]);
    int best_idx = (votes[0] >= votes[1]) ? 0 : 1;

    vector<LightInfo> results;
    if (max_v < vote_threshold) return results;

    LightColor stable = LightColor(best_idx);
    vector<Rect>* active_rois = nullptr;
    Mat* active_mask = nullptr;
    if (stable == RED)   { active_rois = &red_rois;   active_mask = &red_mask; }
    if (stable == GREEN) { active_rois = &green_rois; active_mask = &green_mask; }

    for (auto& r : *active_rois)
    {
        LightInfo info;
        info.box = r;
        info.color = stable;

        Mat roi_bin = (*active_mask)(r);
        ArrowType matched = matchArrow(roi_bin, stable);

        if (matched != CIRCLE)
        {
            info.direction = matched;
        }
        else
        {
            float aspect = (float)r.width / (float)r.height;
            info.direction = (aspect > 0.8f && aspect < 1.2f) ? CIRCLE : matched;
        }
        results.push_back(info);
    }
    return results;
}
