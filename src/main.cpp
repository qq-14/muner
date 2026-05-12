#include "PreProcess.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ============================================================
// 辅助函数：将枚举转换为可读字符串和绘制颜色
// ============================================================
static string colorName(LightColor c)
{
    switch (c)
    {
    case RED:   return "RED";
    case GREEN: return "GREEN";
    default:    return "UNKNOWN";
    }
}

static string arrowName(ArrowType a)
{
    switch (a)
    {
    case CIRCLE:   return "CIRCLE";
    case LEFT:     return "LEFT";
    case RIGHT:    return "RIGHT";
    case STRAIGHT: return "STRAIGHT";
    default:       return "UNKNOWN";
    }
}

static Scalar colorScalar(LightColor c)
{
    switch (c)
    {
    case RED:   return Scalar(0,   0,   255);   // BGR: 红
    case GREEN: return Scalar(0,   255, 0);     // BGR: 绿
    default:    return Scalar(255, 255, 255);
    }
}

// ============================================================
// main：程序入口
// 打开视频 → 逐帧识别 → 绘制结果 → 显示窗口
// ============================================================
int main(int argc, char** argv)
{
    // ---------- 步骤1: 打开视频文件 ----------
    // 优先使用 CMake 定义的 VIDEO_PATH，也支持命令行参数覆盖
    string video_path = VIDEO_PATH;
    if (argc > 1)
        video_path = argv[1];

    VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cerr << "Failed to open video: " << video_path << endl;
        return -1;
    }

    // ---------- 步骤2: 实例化处理器 ----------
    // PreProcess 构造时会自动加载模板图像
    PreProcess processor;

    Mat frame;
    // ---------- 步骤3: 逐帧处理 ----------
    while (cap.read(frame))
    {
        // 调用核心识别流水线，返回当前帧所有检测到的信号灯
        vector<LightInfo> lights = processor.process(frame);

        // ---------- 步骤4: 绘制识别结果 ----------
        for (auto& l : lights)
        {
            // 用对应颜色画矩形框
            rectangle(frame, l.box, colorScalar(l.color), 2);
            // 在框上方写标签，如 "GREEN LEFT"
            string label = colorName(l.color) + " " + arrowName(l.direction);
            putText(frame, label, Point(l.box.x, l.box.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, colorScalar(l.color), 2);
        }

        // ---------- 步骤5: 显示结果 ----------
        imshow("Traffic Light Recognition", frame);

        // 按 ESC 键退出
        if (waitKey(50) == 27) break;
    }

    // ---------- 步骤6: 释放资源 ----------
    cap.release();
    destroyAllWindows();
    return 0;
}
