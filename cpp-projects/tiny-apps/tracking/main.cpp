#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>
#include <sstream>
#include <string>

void help();
void processVideo(std::string videoFilename);
void processImages(std::string dirName);
void mouseHandler(int event, int x, int y, int flags, void *param);

bool selection = false, drawing_box = false;
cv::Rect2d box;
cv::Ptr<cv::Tracker> tracker;

void help()
{
    std::cout
    << "--------------------------------------------------------------------------" << std::endl
    << "Usage:"                                                                     << std::endl
    << "./opencv_exp trackerName {-vid | -img} { <video filename> | <dir name> }"   << std::endl
    << "trackerName can be one of the following: "                                  << std::endl
    << "Boosting, KCF, MedianFlow, MIL, TLD"                                        << std::endl
    << "for example:"                                                               << std::endl
    << "to use video file: ./opencv_exp KCF -vid test.mp4"                          << std::endl
    << "to use image sequence: ./opencv_exp KCF -img test"                          << std::endl
    << "--------------------------------------------------------------------------" << std::endl
    << std::endl;
}

int main(int argc, char* argv[])
{
    // check for the input parameter correctness
    if(argc != 4)
    {
        std::cerr <<"Incorret input list" << std::endl;
        std::cerr <<"exiting..." << std::endl;
        return EXIT_FAILURE;
    }
    // create GUI windows
    cv::namedWindow("Tracking");
    cv::setMouseCallback("Tracking", mouseHandler);
    // tracker method
    if (strcmp(argv[1], "Boosting") == 0)
        tracker = cv::TrackerBoosting::create();
    else if (strcmp(argv[1], "KCF") == 0)
        tracker = cv::TrackerKCF::create();
    else if (strcmp(argv[1], "MedianFlow") == 0)
        tracker = cv::TrackerMedianFlow::create();
    else if (strcmp(argv[1], "MIL") == 0)
        tracker = cv::TrackerMIL::create();
    else if (strcmp(argv[1], "TLD") == 0)
        tracker = cv::TrackerTLD::create();
    else
    {
        //error in reading input parameters
        std::cerr <<"Please, check the input parameters." << std::endl;
        std::cerr <<"Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    // using video or image sequence?
    if (strcmp(argv[2], "-vid") == 0)
        processVideo(argv[3]);
    else if (strcmp(argv[2], "-img") == 0)
        processImages(argv[3]);
    else
    {
        //error in reading input parameters
        std::cerr <<"Please, check the input parameters." << std::endl;
        std::cerr <<"Exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}

void processVideo(std::string videoFilename)
{
    cv::Mat frame, temp;
    cv::VideoCapture cap;
    cap.open(videoFilename);
    if(!cap.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open video file: " << videoFilename << std::endl;
        exit(EXIT_FAILURE);
    }

    cap.read(frame);
    cv::imshow("Tracking", frame);
    while (!selection)
    {
        temp = frame.clone();
        cv::rectangle(temp, box, cv::Scalar(0,0,255), 2);
        cv::imshow("Tracking", temp);
        cv::waitKey(30);
    }
    cv::setMouseCallback("Tracking", NULL);
    tracker->init(frame, box);

    while ((char)cv::waitKey(30) != 'q')
    {
        if(!cap.read(frame))
        {
            std::cerr << "Unable to read next frame." << std::endl;
            std::cerr << "Exiting..." << std::endl;
            break;
        }

        tracker->update(frame, box);
        cv::rectangle(frame, box, cv::Scalar(0,0,255), 2);
        cv::imshow("Tracking", frame);
    }
}

void processImages(std::string dirName)
{
    cv::Mat frame, temp;

    frame = cv::imread(dirName + "/00001.jpg");
    if (frame.empty())
    {
        std::cerr << "Unable to read frame image." << std::endl;
        std::cerr << "Exiting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::imshow("Tracking", frame);
    while ((char) cv::waitKey(30) != 'y')
    {
        temp = frame.clone();
        //cv::rectangle(temp, box, cv::Scalar(0,0,255), 2);
        cv::imshow("Tracking", temp);
        //cv::waitKey(30);
    }
    cv::setMouseCallback("Tracking", NULL);
    //tracker->init(frame, box);

    int numFrame = 2;
    std::stringstream ss;
    std::string fileName;
    while ((char)cv::waitKey(30) != 'q')
    {
        ss.str(""); ss << numFrame;
        frame = cv::Mat();
        if (numFrame < 10)
            fileName = dirName + "/0000" + ss.str() + ".jpg";
        else if (numFrame < 100)
            fileName = dirName + "/000" + ss.str() + ".jpg";
        else if (numFrame < 1000)
            fileName = dirName + "/00" + ss.str() + ".jpg";
        else if (numFrame < 10000)
            fileName = dirName + "/0" + ss.str() + ".jpg";
        else
            fileName = dirName + "/" + ss.str() + ".jpg";
        frame = cv::imread(fileName);
        if(frame.empty())
        {
            std::cerr << "Unable to read next frame." << std::endl;
            std::cerr << "Exiting..." << std::endl;
            break;
        }

        // tracker->update(frame, box);
        // cv::rectangle(frame, box, cv::Scalar(0,0,255), 2);
        cv::imshow("Tracking", frame);
        numFrame ++;
    }
}

void mouseHandler(int event, int x, int y, int flags, void *param)
{
    switch (event)
    {
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box)
        {
            box.width = x - box.x;
            box.height = y - box.y;
        }
        break;
    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cv::Rect(x, y, 0, 0);
        break;
    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if (box.width < 0)
        {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0)
        {
            box.y += box.height;
            box.height *= -1;
        }
        selection = true;
        break;
    }
}
