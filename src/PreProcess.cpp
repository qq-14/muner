#include "PreProcess.hpp"
#include <opencv2/opencv.hpp>

cv::Mat PreProcess::process(const cv::Mat& frame){
    // 对图像使用高斯滤波进行降噪
    cv::Mat blur;
    cv::GaussianBlur(frame, blur, cv::Size(gaussian_k, gaussian_k), 0);
    // 将图像转为hsv
    cv::Mat hsv;
    cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);
    // 对处理过的hsv图像进行颜色分割(提取黄色，红色，绿色)
    // 对两个红色区间进行掩码合并
    Mat mask1, mask2, red_mask;
    inRange(hsv, low_red1, high_red1, mask1);
    inRange(hsv, low_red2, high_red2, mask2);

    red_mask = mask1 | mask2;

    // 绿色掩码合并
    Mat green_mask;
    inRange(hsv, low_green, high_green, green_mask);

    // 黄色掩码合并
    Mat yellow_mask;
    inRange(hsv, low_yellow, high_yellow, yellow_mask);

    // 对图像进行形态学去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(morph_k, morph_k));
    morphologyEx(red_mask, red_mask, MORPH_OPEN, kernel);
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);

   
}