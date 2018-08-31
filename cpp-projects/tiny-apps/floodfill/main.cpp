#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat image, mask;

void myMouseEvent(int event, int x, int y, int flags, void *param);

int main()
{
    image = cv::imread("d:/develop/dl/projects/resources/images/person-dog-horse.jpg");
    mask  = cv::Mat(image.rows+2, image.cols+2, CV_8UC1, cv::Scalar::all(0));
    cv::namedWindow("Flood Fill Demo");
    cv::namedWindow("Mask");

    cv::imshow("Flood Fill",image);
    cv::setMouseCallback("Flood Fill",myMouseEvent);

    while ((char)cv::waitKey(20) != 'q')
    {
        cv::imshow("Flood Fill",image);
        cv::imshow("Mask",mask);
        //cv::imwrite("FloodFill.jpg",image);
        //cv::imwrite("FloodFillMask.jpg",mask);
    }

    return 0;
}

void myMouseEvent(int event, int x, int y, int flags, void *param)
{
    int floodFill_flag = 8 | (255<<8);  // consider 4 nearest neighbours and fill the mask with a value of 255
    cv::Mat Mask = cv::Mat(image.rows+2, image.cols+2,
                           CV_8UC1, cv::Scalar::all(0));

    if (event == CV_EVENT_LBUTTONDOWN)
    {
        cv::Point seed = cv::Point(x,y);
        cv::floodFill(image, Mask, seed,
                      cv::Scalar(255,0,0), 0,
                      cv::Scalar(10,10,10), cv::Scalar(10,10,10),
                      floodFill_flag);
        cv::add(mask, Mask, mask);
    }
}
