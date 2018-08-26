#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
float k = 15;
float lambda = 0.25;
int N = 20;

void anisotropy_demo(Mat &image, Mat &result);
int main(int argc, char** argv) {
    Mat src = imread("../../../datas/face3.jpg");
    if (src.empty()) {
        printf("could not load image...\n");
        return -1;
    }
    namedWindow("input image");
    imshow("input image", src);
    vector<Mat> mv;
    vector<Mat> results;
    split(src, mv);
    for (int n = 0; n < mv.size(); n++) {
        Mat m = Mat::zeros(src.size(), CV_32FC1);
        mv[n].convertTo(m, CV_32FC1);
        results.push_back(m);
    }

    int w = src.cols;
    int h = src.rows;
    Mat copy = Mat::zeros(src.size(), CV_32FC1);
    for (int i = 0; i < N; i++) {
        anisotropy_demo(results[0], copy);
        copy.copyTo(results[0]);

        anisotropy_demo(results[1], copy);
        copy.copyTo(results[1]);

        anisotropy_demo(results[2], copy);
        copy.copyTo(results[2]);

    }
    Mat output;
    normalize(results[0], results[0], 0, 255, NORM_MINMAX);
    normalize(results[1], results[1], 0, 255, NORM_MINMAX);
    normalize(results[2], results[2], 0, 255, NORM_MINMAX);

    results[0].convertTo(mv[0], CV_8UC1);
    results[1].convertTo(mv[1], CV_8UC1);
    results[2].convertTo(mv[2], CV_8UC1);

    Mat dst;
    merge(mv, dst);

    imshow("result", dst);
    //imwrite("D:/result.png", dst);
    waitKey(0);
    return 0;
}

void anisotropy_demo(Mat &image, Mat &result) {
    int width = image.cols;
    int height = image.rows;

    // 四邻域梯度
    float n = 0, s = 0, e = 0, w = 0;
    // 四邻域系数
    float nc = 0, sc = 0, ec = 0, wc = 0;
    float k2 = k*k;
    for (int row = 1; row < height -1; row++) {
        for (int col = 1; col < width -1; col++) {
            // gradient
            n = image.at<float>(row - 1, col) - image.at<float>(row, col);
            s = image.at<float>(row + 1, col) - image.at<float>(row, col);
            e = image.at<float>(row, col - 1) - image.at<float>(row, col);
            w = image.at<float>(row, col + 1) - image.at<float>(row, col);
            nc = exp(-n*n / k2);
            sc = exp(-s*s / k2);
            ec = exp(-e*e / k2);
            wc = exp(-w*w / k2);
            result.at<float>(row, col) = image.at<float>(row, col) + lambda*(n*nc + s*sc + e*ec + w*wc);
        }
    }
}
