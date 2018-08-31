#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>

int main()
{
    cv::Mat image = cv::imread("d:/develop/dl/projects/resources/images/pedestrain.jpg");
    cv::namedWindow("Original Image");
    cv::imshow("Original Image",image);

    std::vector<cv::Rect> locations;
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    //cv::GaussianBlur(image,image,cv::Size(3,3),5);
    hog.detectMultiScale(image,locations);

    cv::Mat result = image.clone();
    for (int i = 0; i<locations.size(); i++)
    {
        cv::rectangle(result,locations.at(i),
                      cv::Scalar(0,0,255),2);
    }

    cv::namedWindow("Detected Objects");
    cv::imshow("Detected Objects",result);
//    cv::imwrite("result.jpg", result);
    std::cout << "Number of detections:" << locations.size() << std::endl;

    cv::waitKey();
    return 0;
}
