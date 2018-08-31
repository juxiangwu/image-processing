#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#ifdef WINDOWS
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#endif

#include "Blob.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    capVideo.open("D:/Develop/DL/projects/resources/videos/768x576.avi");

    if (!capVideo.isOpened()) {                                                 // if unable to open video file
        std::cout << "\nerror reading video file" << std::endl << std::endl;      // show error message
        #ifdef WINDOWS
        _getch();                    // it may be necessary to change or remove this line if not using Windows
        #endif
        return(0);                                                              // and exit program
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "\nerror: video file must have at least two frames";
        #ifdef WINDOWS
        _getch();
        #endif
        return(0);
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    char chCheckForEscKey = 0;

    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        std::vector<Blob> blobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

        cv::imshow("imgThresh", imgThresh);

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

        cv::dilate(imgThresh, imgThresh, structuringElement5x5);
        cv::dilate(imgThresh, imgThresh, structuringElement5x5);
        cv::erode(imgThresh, imgThresh, structuringElement5x5);

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

        cv::imshow("imgContours", imgContours);

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.boundingRect.area() > 100 &&
                possibleBlob.dblAspectRatio >= 0.2 &&
                possibleBlob.dblAspectRatio <= 1.2 &&
                possibleBlob.boundingRect.width > 15 &&
                possibleBlob.boundingRect.height > 20 &&
                possibleBlob.dblDiagonalSize > 30.0) {
                blobs.push_back(possibleBlob);
            }
        }

        cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);
        
        convexHulls.clear();

        for (auto &blob : blobs) {
            convexHulls.push_back(blob.contour);
        }

        cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

        cv::imshow("imgConvexHulls", imgConvexHulls);

        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        for (auto &blob : blobs) {                                                  // for each blob
            cv::rectangle(imgFrame2Copy, blob.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
            cv::circle(imgFrame2Copy, blob.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
        }

        cv::imshow("imgFrame2Copy", imgFrame2Copy);

                // now we prepare for the next iteration

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {       // if there is at least one more frame
            capVideo.read(imgFrame2);                            // read it
        }
        else {                                                  // else
            std::cout << "end of video\n";                      // show end of video message
            break;                                              // and jump out of while loop
        }

        chCheckForEscKey = cv::waitKey(1);      // get key press in case user pressed esc

    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

    return(0);
}