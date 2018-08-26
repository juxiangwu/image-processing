//添加这个宏避免中文乱码
#if _MSC_VER >= 1600
# pragma execution_character_set(“utf-8”)
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>

#include <iostream>

using namespace std;
using namespace cv::freetype;

int main()
{
    cv::Mat src = cv::imread("../../../datas/face3.jpg");
    if (src.empty()) {
        printf("could not load image...\n");
        return -1;
    }
    const string font_name = "../../../datas/fonts/msyh.ttf";
    cv::Ptr<FreeType2> ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(font_name,0);
    ft2->setSplitNumber(4);
    const string text("美女");
    ft2->putText(src,text,
                 cv::Point2d(100,100),
                 20,cv::Scalar(0,255,0),
                 -1,cv::LINE_AA,false);

    cv::imshow("src",src);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
