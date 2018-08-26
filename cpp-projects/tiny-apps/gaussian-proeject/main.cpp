#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
//高斯反向投影
int main(int argc, char** argv) {
    Mat src = imread("../../../datas/f3.jpg");
    Mat model = imread("../../../datas/f3-model.jpg");
    if (src.empty() || model.empty()) {
        printf("could not load image...\n");
        return -1;
    }
    imshow("input image", src);

    Mat R = Mat::zeros(model.size(), CV_32FC1);
    Mat G = Mat::zeros(model.size(), CV_32FC1);
    int r = 0, g = 0, b = 0;
    float sum = 0;
    for (int row = 0; row < model.rows; row++) {
        uchar* current = model.ptr<uchar>(row);
        for (int col = 0; col < model.cols; col++) {
            b = *current++;
            g = *current++;
            r = *current++;
            sum = b + g + r;
            R.at<float>(row, col) = r / sum;
            G.at<float>(row, col) = g / sum;
        }
    }

    Mat mean, stddev;
    double mr, devr;
    double mg, devg;
    meanStdDev(R, mean, stddev);
    mr = mean.at<double>(0, 0);
    devr = mean.at<double>(0, 0);

    meanStdDev(G, mean, stddev);
    mg = mean.at<double>(0, 0);
    devg = mean.at<double>(0, 0);

    int width = src.cols;
    int height = src.rows;

    float pr = 0, pg = 0;
    Mat result = Mat::zeros(src.size(), CV_32FC1);
    for (int row = 0; row < height; row++) {
        uchar* currentRow = src.ptr<uchar>(row);
        for (int col = 0; col < width; col++) {
            b = *currentRow++;
            g = *currentRow++;
            r = *currentRow++;
            sum = b + g + r;
            float red = r / sum;
            float green = g / sum;
            pr = (1 / (devr*sqrt(2 * CV_PI)))*exp(-(pow((red - mr), 2)) / (2 * pow(devr, 2)));
            pg = (1 / (devg*sqrt(2 * CV_PI)))*exp(-(pow((green - mg),2)) / (2 * pow(devg, 2)));
            sum = pr*pg;
            result.at<float>(row, col) = sum;
        }
    }

    Mat img(src.size(), CV_8UC1);
    normalize(result, result, 0, 255, NORM_MINMAX);
    result.convertTo(img, CV_8U);
    Mat segmentation;
    src.copyTo(segmentation, img);

    imshow("backprojection demo", img);
    imshow("segmentation demo", segmentation);

    waitKey(0);
    return 0;
}
