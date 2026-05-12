#include "PreProcess.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>

using namespace cv;
using namespace std;

// ============================================================
// 构造函数：初始化 HSV 阈值、形态学核，并加载模板图像
// ============================================================
PreProcess::PreProcess() : gaussian_k(5), morph_k(7), min_area(120), max_area(5000), match_threshold(0.6),
                            history_max(10), vote_threshold(7)
{
    // 步骤1: 设置红色 HSV 区间 —— 红色在 HSV 中跨越 0° 和 180°，需两个区间
    low_red1  = Scalar(0,   120, 70);
    high_red1 = Scalar(10,  255, 255);
    low_red2  = Scalar(170, 120, 70);
    high_red2 = Scalar(180, 255, 255);

    // 步骤2: 设置绿色 HSV 区间
    low_green  = Scalar(35, 50, 50);
    high_green = Scalar(90, 255, 255);

    // 步骤3: 设置黄色 HSV 区间
    // 实测视频中固定黄光区域 HSV ≈ (19~20, 150~158, 205~218)
    // 缩紧阈值以区分"灯亮"与"外壳反射"，同时降低 S 下限排除偏色
    low_yellow  = Scalar(20, 130, 200);
    high_yellow = Scalar(32, 255, 255);

    // 步骤4: 创建形态学操作核 (5×5 矩形)
    kernel = getStructuringElement(MORPH_RECT, Size(morph_k, morph_k));

    // 步骤5: 从磁盘加载箭头模板（二值图）
    loadTemplates();
}

// ============================================================
// loadTemplates：遍历 Tamplate/ 下的 red/green/yellow 文件夹，
//   读取全部 _bin.png 二值图，根据文件名判断箭头方向并存入 templates 字典
// ============================================================
void PreProcess::loadTemplates()
{
    string base = TEMPLATE_DIR;
    vector<pair<LightColor, string>> color_map = {
        {RED,   base + "/red"},
        {GREEN, base + "/green"},
        {YELLO, base + "/yellow"}
    };

    for (auto& [color, dir_path] : color_map)
    {
        if (!filesystem::exists(dir_path)) continue;
        for (auto& entry : filesystem::directory_iterator(dir_path))
        {
            string ext = entry.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".png") continue;

            // 以灰度图读入模板（二值图只需单通道）
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            // 从文件名推断箭头方向（如 R_left_bin.png → LEFT）
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

// ============================================================
// createMask：在 HSV 图像上执行 inRange 阈值分割，返回二值掩码
// ============================================================
Mat PreProcess::createMask(const Mat& hsv, const Scalar& low, const Scalar& high)
{
    Mat mask;
    inRange(hsv, low, high, mask);
    imshow("binary", mask);
    return mask;
}

// ============================================================
// cleanMask：对二值掩码做形态学开运算（去噪点）+ 闭运算（填孔洞）
// ============================================================
void PreProcess::cleanMask(Mat& mask)
{
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
}

// ============================================================
// findLightROIs：在二值掩码上找外轮廓，过滤面积后返回所有包围盒
// ============================================================
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

// ============================================================
// decideActiveColor：按各颜色 ROI 数量 + 黄色亮度权重 判定
//   若黄色 ROI 中心 V 值 ≥ 230（真实点亮而非反射），权重 ×3
// ============================================================
LightColor PreProcess::decideActiveColor(const vector<Rect>& red_rois,
                                          const vector<Rect>& green_rois,
                                          const vector<Rect>& yellow_rois)
{
    double r = red_rois.size(), g = green_rois.size(), y = yellow_rois.size();

    if (y > 0 && !last_hsv.empty())
    {
        int cx = yellow_rois[0].x + yellow_rois[0].width / 2;
        int cy = yellow_rois[0].y + yellow_rois[0].height / 2;
        if (cx >= 0 && cy >= 0 && cx < last_hsv.cols && cy < last_hsv.rows)
        {
            uchar v = last_hsv.at<Vec3b>(cy, cx)[2];
            if (v >= 230) y *= 3.0;
        }
    }

    double m = max({r, g, y});
    if (m == 0) return RED;
    if (r == m) return RED;
    if (g == m) return GREEN;
    return YELLO;
}

// ============================================================
// matchArrow：对 ROI 二值图做模板匹配
//   1. 将 ROI resize 到模板尺寸
//   2. matchTemplate 使用 TM_CCOEFF_NORMED（归一化相关系数）
//   3. 取最高分 > match_threshold 的方向，否则返回 CIRCLE
// ============================================================
ArrowType PreProcess::matchArrow(const Mat& roi, LightColor color)
{
    auto it = templates.find(color);
    if (it == templates.end() || it->second.empty()) return CIRCLE;

    double best_val = -1.0;
    ArrowType best_type = CIRCLE;

    for (auto& [tmpl, type] : it->second)
    {
        // 将 ROI 缩放到与模板一致
        Mat resized;
        resize(roi, resized, tmpl.size());

        // 归一化相关系数匹配
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

// ============================================================
// process：核心处理函数 —— 对一帧图像执行完整识别流水线
// 返回 vector<LightInfo>，包含每个检测到的信号灯信息
// ============================================================
vector<LightInfo> PreProcess::process(const Mat& frame)
{
    // ---------- 步骤1: 预处理 ----------
    // 高斯模糊降噪，减少 HSV 阈值分割时的噪点
    Mat blurred, hsv;
    GaussianBlur(frame, blurred, Size(gaussian_k, gaussian_k), 0);
    // BGR → HSV 转换（H 色调 / S 饱和度 / V 明度，对光照变化更鲁棒）
    cvtColor(blurred, hsv, COLOR_BGR2HSV);

    // 保存 HSV 供 decideActiveColor 读取 V 值
    last_hsv = hsv;

    // ---------- 步骤2: HSV 颜色分割 ----------
    // 分别生成红 / 绿 / 黄三色的二值掩码
    Mat rm1 = createMask(hsv, low_red1, high_red1);
    Mat rm2 = createMask(hsv, low_red2, high_red2);
    red_mask = rm1 | rm2;       // 红色取两个区间之并集
    green_mask = createMask(hsv, low_green, high_green);
    yellow_mask = createMask(hsv, low_yellow, high_yellow);

    // ---------- 步骤3: 形态学去噪 ----------
    // 开运算去除孤立白点，闭运算填补信号灯内部小黑洞
    cleanMask(red_mask);
    cleanMask(green_mask);
    cleanMask(yellow_mask);

    // ---------- 步骤4: 轮廓检测 + 面积过滤 ----------
    vector<vector<Point>> red_all, green_all, yellow_all;
    findContours(red_mask, red_all, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(green_mask, green_all, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(yellow_mask, yellow_all, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 按面积筛选并保存有效轮廓
    red_contours.clear(); green_contours.clear(); yellow_contours.clear();
    for (auto& c : red_all)    if (contourArea(c) >= min_area && contourArea(c) <= max_area) red_contours.push_back(c);
    for (auto& c : green_all)  if (contourArea(c) >= min_area && contourArea(c) <= max_area) green_contours.push_back(c);
    for (auto& c : yellow_all) if (contourArea(c) >= min_area && contourArea(c) <= max_area) yellow_contours.push_back(c);

    // ---------- 步骤5: ROI 截取 ----------
    // 从面积过滤后的掩码上提取外接矩形作为候选区域
    auto red_rois   = findLightROIs(red_mask);
    auto green_rois = findLightROIs(green_mask);
    auto yellow_rois = findLightROIs(yellow_mask);

    // ---------- 步骤6: 活跃颜色判定 ----------
    // 当前帧中哪个颜色的候选区最多，就认为该颜色是激活的信号灯
    LightColor active = decideActiveColor(red_rois, green_rois, yellow_rois);

    // ---------- 步骤7: 多帧稳定（滑动窗口投票）----------
    // 将当前帧的活跃颜色推入历史队列，统计最近 N 帧中每种颜色的票数
    history.push_back(active);
    if ((int)history.size() > history_max) history.pop_front();

    int votes[3] = {0};
    for (auto c : history) votes[c]++;

    int max_v = votes[0], best_idx = 0;
    for (int i = 1; i < 3; i++)
        if (votes[i] > max_v) { max_v = votes[i]; best_idx = i; }

    vector<LightInfo> results;
    // 票数未达阈值则返回空（避免闪烁误报）
    if (max_v < vote_threshold) return results;

    LightColor stable = LightColor(best_idx);
    vector<Rect>* active_rois = nullptr;
    Mat* active_mask = nullptr;
    if (stable == RED)   { active_rois = &red_rois;   active_mask = &red_mask; }
    if (stable == GREEN) { active_rois = &green_rois; active_mask = &green_mask; }
    if (stable == YELLO) { active_rois = &yellow_rois; active_mask = &yellow_mask; }

    // ---------- 步骤8: 逐 ROI 分析 —— 先模板匹配，宽高比辅助判断 ----------
    for (auto& r : *active_rois)
    {
        LightInfo info;
        info.box = r;
        info.color = stable;

        // ---------- 步骤9: 箭头模板匹配（所有 ROI 都先尝试匹配）----------
        // 从对应颜色的二值掩码中裁剪 ROI 区域
        Mat roi_bin = (*active_mask)(r);
        // 与预存箭头模板做匹配，返回最佳方向
        ArrowType matched = matchArrow(roi_bin, stable);

        if (matched != CIRCLE)
        {
            // 模板匹配成功识别出箭头方向
            info.direction = matched;
        }
        else
        {
            // 无箭头模板匹配成功 → 用宽高比辅助判断是否为圆灯
            float aspect = (float)r.width / (float)r.height;
            info.direction = (aspect > 0.8f && aspect < 1.2f) ? CIRCLE : matched;
        }
        results.push_back(info);
    }
    return results;
}
