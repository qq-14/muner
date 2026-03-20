#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct miner {
    float L; // 长
    float W; // 宽
    float Center; //中心点
    vector<Point2f> len_points; //长边中心 
};
