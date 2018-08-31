#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
    cv::Mat imageA = cv::imread("d:/develop/dl/projects/resources/images/person-dog-horse.jpg");
    cv::cvtColor(imageA, imageA, CV_BGR2GRAY);

    // get the filter kernel
    cv::Mat imageBx, imageBy;
    cv::getDerivKernels(imageBx, imageBy,1,1,3,true);
    cv::Mat imageB = imageBx;

    // get the size of the DFT
    // Based on the knowledge of DSP, the size of DFT must be not less than lthe size of linear convolution.
    int dft_M = cv::getOptimalDFTSize(imageA.rows + imageB.rows -1);
    int dft_N = cv::getOptimalDFTSize(imageA.cols + imageB.cols -1);

    // calclute the DFT of image & filter
    cv::Mat dft_A = cv::Mat::zeros(dft_M, dft_N, CV_32FC1);
    cv::Mat dft_B = cv::Mat::zeros(dft_M, dft_N, CV_32FC1);
    //note that dft_A_part is the SHALLOW copy of  dft_A; same as dft_B_part
    cv::Mat dft_A_part = dft_A(cv::Rect(0,0, imageA.cols, imageA.rows));
    cv::Mat dft_B_part = dft_B(cv::Rect(0,0, imageB.cols, imageB.rows));
    imageA.convertTo(dft_A_part, dft_A_part.type());
    imageB.convertTo(dft_B_part, dft_B_part.type());
    cv::dft(dft_A, dft_A, 0, imageA.rows);
    cv::dft(dft_B, dft_B, 0, imageB.rows);

    // calclute multipization in frequency domain
    cv::Mat dft_result;
    cv::mulSpectrums(dft_A, dft_B, dft_result, cv::DFT_ROWS, false);

    // calclute IDFT of the multipization result
    cv::idft(dft_result, dft_result, cv::DFT_SCALE,
             imageA.rows + imageB.rows - 1);

    cv::Mat result;
    result.create(imageA.rows + imageB.rows -1,
                  imageA.cols + imageB.cols -1,
                  dft_result.type());
    result = dft_result(cv::Rect(0,0,result.cols, result.rows));
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);

    cv::namedWindow("Image");
    cv::imshow("Image", imageA);
    cv::namedWindow("Filter Result");
    cv::imshow("Filter Result",result);
    //cv::imwrite("result.jpg",result);

    cv::waitKey(0);
    return 0;
}
