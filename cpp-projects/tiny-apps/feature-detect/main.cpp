#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct SIFTDetector
{
    Ptr<SIFT> sift;
    SIFTDetector(int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6)
    {
        sift = SIFT::create(nfeatures, nOctaveLayers,
            contrastThreshold, edgeThreshold, sigma);
    }
    void operator()(const Mat& inputImage, vector<KeyPoint>& pts, const Mat& mask = Mat())
    {
        sift->detect(inputImage, pts, mask);
    }
    void operator()(const Mat& in, vector<KeyPoint>& pts, Mat& descriptors,
        const Mat& mask = Mat(), bool useProvided = false)
    {
        sift->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

struct SURFDetector
{
    Ptr<SURF> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    void operator()(const Mat& inputImage, vector<KeyPoint>& pts, const Mat& mask = Mat())
    {
        surf->detect(inputImage, pts, mask);
    }
    void operator()(const Mat& in, vector<KeyPoint>& pts, Mat& descriptors,
        const Mat& mask = Mat(), bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

int main()
{
    // Load Image
    Mat image = imread("d:/develop/dl/projects/resources/images/dog-cycle-car.png");
    cvtColor(image,image,CV_BGR2GRAY);

    // Detector SIFT Features
    SIFTDetector sift_detector;
    vector<KeyPoint> sift_keypoints;
    //Mat sift_descriptors;
    //detector(image, sift_keypoints, sift_descriptors);
    sift_detector(image, sift_keypoints);

    // Detector SURF Features
    SURFDetector surf_detector;
    vector<KeyPoint> surf_keypoints;
    //Mat surf_descriptors;
    //detector(image, surf_keypoints, surf_descriptors);
    surf_detector(image, surf_keypoints);

    Mat imageForDraw;
    // Display SIFT Features
    drawKeypoints(image, sift_keypoints, imageForDraw, Scalar::all(255));
    namedWindow("Show SIFT features");
    imshow("Show SIFT features", imageForDraw);
//	cv::imwrite("SIFT.jpg", imageForDraw);

    // Display SUFR Features
    //clear
    imageForDraw = Mat();
    drawKeypoints(image, surf_keypoints, imageForDraw, Scalar::all(255));
    namedWindow("Show SURF features");
    imshow("Show SURF features", imageForDraw);
//	cv::imwrite("SURF.jpg", imageForDraw);

    waitKey();
    return 0;
}
