#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
    cv::Mat image = cv::imread("d:/develop/dl/projects/resources/images/lsd-test.jpg");
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    cv::Mat grayI, result, result_filtered;
    cv::cvtColor(image, grayI, CV_BGR2GRAY);
    result = image.clone();
    result_filtered = image.clone();
    std::vector<cv::Vec4f> lines;

    cv::Ptr<cv::LineSegmentDetector> lineDetecor =
        cv::createLineSegmentDetector();
    lineDetecor->detect(grayI, lines);
    lineDetecor->drawSegments(result, lines);

    for (int i = 0; i < lines.size(); i++)
    {
        cv::Vec4f L = lines[i];
        double x1 = L[0], y1 = L[1];
        double x2 = L[2], y2 = L[3];
        if (std::sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) > 30)
            cv::line(result_filtered, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255));
    }

    cv::namedWindow("Detection Results");
    cv::imshow("Detection Results", result);
    cv::namedWindow("Filtered Results");
    cv::imshow("Filtered Results", result_filtered);
    //cv::imwrite("result.jpg",result);
    //cv::imwrite("result_filtered.jpg",result_filtered);

    cv::waitKey(0);
    return 0;
}
