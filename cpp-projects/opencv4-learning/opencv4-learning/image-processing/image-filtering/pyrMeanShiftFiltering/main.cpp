#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat src = cv::imread("../../../../../../datas/city.jpg");
    if(src.empty()){
        std::cerr << "cannot open image" << std::endl;
        return -1;
    }
    cv::Mat dst;
    int spatialRad = 50;  //空间窗口大小
    int colorRad = 50;   //色彩窗口大小
    int maxPyrLevel = 2;  //金字塔层数

    cv::pyrMeanShiftFiltering( src, dst, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波

    cv::RNG rng = cv::theRNG();
    cv::Mat mask( dst.rows+2, dst.cols+2, CV_8UC1, cv::Scalar::all(0) );  //掩模
    cv::imshow("src",src);
    cv::imshow("dst",dst);

    for( int y = 0; y < dst.rows; y++ )
        {
            for( int x = 0; x < dst.cols; x++ )
            {
                if( mask.at<uchar>(y+1, x+1) == 0 )  //非0处即为1，表示已经经过填充，不再处理
                {
                    cv::Scalar newVal( rng(256), rng(256), rng(256) );
                    cv::floodFill( dst, mask, cv::Point(x,y), newVal, 0, cv::Scalar::all(5), cv::Scalar::all(5) ); //执行漫水填充
                }
            }
        }

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
