#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace std;
static int fps = 0;
static double f = (1000 / getTickFrequency());
static int64 startTime;
static int cnt, oldcnt;

void startFPS() {
    startTime = cv::getTickCount();
    cnt = oldcnt = 0;
}

void stopFPS() {
    int64 nowTime, diffTime;
    nowTime = cv::getTickCount();
    diffTime = (int)((nowTime - startTime) * f);
    cout << "time= " << diffTime << endl;
}

void tickFPS() {
    int64 nowTime, diffTime;
    int sec = 5;
    nowTime = cv::getTickCount();
    diffTime = (int)((nowTime - startTime) * f);
    if (diffTime >= sec * 1000) {
        startTime = nowTime;
        fps = (cnt - oldcnt) / (float)sec;
        oldcnt = cnt;
        cout << "fps=" << fps << "\n";
    }
    cnt++;
}
