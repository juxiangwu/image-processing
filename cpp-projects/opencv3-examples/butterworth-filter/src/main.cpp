// Author: Alessandro Gentilini, 2013, based on a work by Toby Breckon,
// originally found at http://public.cranfield.ac.uk/c5354/teaching/dip/opencv/lecture_demos/c++/butterworth_lowpass.cpp
// n ow at https://github.com/tobybreckon/cpp-examples-ipcv/blob/master/butterworth_lowpass.cpp

// Example : apply butterworth low pass filtering to input image/video
// usage: prog {<image_name> | <video_name>}

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

//#include <cv.h>         // open cv general include file
//#include <highgui.h>    // open cv GUI include file
#include <opencv2/opencv.hpp>
#include <iostream>     // standard C++ I/O
#define _USE_MATH_DEFINES
#include <math.h>
using namespace cv; // OpenCV API is in the C++ "cv" namespace
using namespace std;

// #define M_PI 3.1415926
/******************************************************************************/
// setup the cameras properly based on OS platform

// 0 in linux gives first camera for v4l
//-1 in windows gives first device or user dialog selection

#ifdef linux
#define CAMERA_INDEX 0
#else
#define CAMERA_INDEX -1
#endif
/******************************************************************************/
// Rearrange the quadrants of a Fourier image so that the origin is at
// the image center

void shiftDFT(Mat &fImage )
{
    Mat tmp, q0, q1, q2, q3;

    // first crop the image, if it has an odd number of rows or columns

    fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

    int cx = fImage.cols / 2;
    int cy = fImage.rows / 2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center

    q0 = fImage(Rect(0, 0, cx, cy));
    q1 = fImage(Rect(cx, 0, cx, cy));
    q2 = fImage(Rect(0, cy, cx, cy));
    q3 = fImage(Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/******************************************************************************/
// return a floating point spectrum magnitude image scaled for user viewing
// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
// rearrange - perform rearrangement of DFT quadrants if true

// return value - pointer to output spectrum magnitude image scaled for user viewing

Mat create_spectrum_magnitude_display(Mat &complexImg, bool rearrange)
{
    Mat planes[2];

    // compute magnitude spectrum (N.B. for display)
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat mag = (planes[0]).clone();
    mag += Scalar::all(1);
    log(mag, mag);

    if (rearrange)
    {
        // re-arrange the quaderants
        shiftDFT(mag);
    }

    normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;

}
/******************************************************************************/

// create a 2-channel butterworth low-pass filter with radius D, order n
// (assumes pre-aollocated size of dft_Filter specifies dimensions)

void create_butterworth_lowpass_filter(Mat &dft_Filter, int D, int n, int W)
{
    Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
    double radius;

    // based on the forumla in the IP notes (p. 130 of 2009/10 version)
    // see also HIPR2 on-line

    for (int i = 0; i < dft_Filter.rows; i++)
    {
        for (int j = 0; j < dft_Filter.cols; j++)
        {
            radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));

            // Butterworth low pass:
            // tmp.at<float>(i,j) = (float)
            //                        ( 1 / (1 + pow((double) (radius /  D), (double) (2 * n))));

            // Butterworth band reject, page 244, paragraph 5.4.1, Gonzalez Woods, "Digital Image Processing 2nd Edition"
            // D(u,v) -> radius
            // D_0 -> D
            tmp.at<float>(i, j) = (float)
                                  ( 1 / (1 + pow((double) (radius * W) / ( pow((double)radius, 2) - D * D ), (double) (2 * n))));
        }
    }

    Mat toMerge[] = {tmp, tmp};
    merge(toMerge, 2, dft_Filter);
}

/******************************************************************************/

int main( int argc, char **argv )
{

    Mat img, imgGray, imgOutput;  // image object(s)
    VideoCapture cap(0); // capture object

    Mat padded;       // fourier image objects and arrays
    Mat complexImg, filter, filterOutput;
    Mat planes[2], mag;

    int N, M; // fourier image sizes

    int radius = 20;              // low pass filter parameter
    int order = 2;                // low pass filter parameter
    int width = 3;

    const string originalName = "Input Image (grayscale)"; // window name
    const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
    const string lowPassName = "Butterworth Low Pass Filtered (grayscale)"; // window name
    const string filterName = "Filter Image"; // window nam

    bool keepProcessing = true;   // loop control flag
    int  key;                     // user input
    int  EVENT_LOOP_DELAY = 40;   // delay for GUI window
    // 40 ms equates to 1000ms/25fps = 40ms per frame

    // if command line arguments are provided try to read image/video_name
    // otherwise default to capture from attached H/W camera

    if (
        ( argc == 2 && (!(img = imread( argv[1], CV_LOAD_IMAGE_COLOR)).empty())) ||
        ( argc == 2 && (cap.open(argv[1]) == true )) ||
        ( argc != 2 && (cap.open(CAMERA_INDEX) == true))
    )
    {
        // create window object (use flag=0 to allow resize, 1 to auto fix size)

        namedWindow(originalName, 0);
        namedWindow(spectrumMagName, 0);
        namedWindow(lowPassName, 0);
        namedWindow(filterName, 0);

        // if capture object in use (i.e. video/camera)
        // get image from capture object

        if (cap.isOpened())
        {

            cap >> img;
            if (img.empty())
            {
                if (argc == 2)
                {
                    std::cerr << "End of video file reached" << std::endl;
                }
                else
                {
                    std::cerr << "ERROR: cannot get next fram from camera"
                              << std::endl;
                }
                exit(0);
            }

        }

        // setup the DFT image sizes

        M = getOptimalDFTSize( img.rows );
        N = getOptimalDFTSize( img.cols );

        // add adjustable trackbar for low pass filter threshold parameter

        createTrackbar("Radius", lowPassName, &radius, (min(M, N) / 2));
        createTrackbar("Order", lowPassName, &order, 10);
        createTrackbar("Width", lowPassName, &width, (min(M, N) / 2));

        // start main loop

        while (keepProcessing)
        {

            // if capture object in use (i.e. video/camera)
            // get image from capture object

            if (cap.isOpened())
            {

                cap >> img;
                if (img.empty())
                {
                    if (argc == 2)
                    {
                        std::cerr << "End of video file reached" << std::endl;
                    }
                    else
                    {
                        std::cerr << "ERROR: cannot get next fram from camera"
                                  << std::endl;
                    }
                    exit(0);
                }

            }

            // ***

            // convert input to grayscale

            cvtColor(img, imgGray, CV_BGR2GRAY);

            // setup the DFT images

            copyMakeBorder(imgGray, padded, 0, M - imgGray.rows, 0,
                           N - imgGray.cols, BORDER_CONSTANT, Scalar::all(0));
            planes[0] = Mat_<float>(padded);
            planes[1] = Mat::zeros(padded.size(), CV_32F);

            merge(planes, 2, complexImg);

            // do the DFT

            dft(complexImg, complexImg);

            // construct the filter (same size as complex image)

            filter = complexImg.clone();
            create_butterworth_lowpass_filter(filter, radius, order, width);

            // apply filter
            shiftDFT(complexImg);
            mulSpectrums(complexImg, filter, complexImg, 0);
            shiftDFT(complexImg);

            // create magnitude spectrum for display

            mag = create_spectrum_magnitude_display(complexImg, true);

            // do inverse DFT on filtered image

            idft(complexImg, complexImg);

            // split into planes and extract plane 0 as output image

            cv::Mat myplanes[2];
            split(complexImg, myplanes);
            double minimum = -1;
            double maximum = -1;
            cv::Point minloc(-1, -1), maxloc(-1, -1);
            minMaxLoc(myplanes[0], &minimum, &maximum, &minloc, &maxloc);
            std::cout << "min=" << minimum << "@" << minloc << "\tmax=" << maximum << "@" << maxloc << "\n";
            //normalize(myplanes[0], imgOutput, 0, 1, CV_MINMAX);
            imgOutput = myplanes[0];

            // do the same with the filter image

            split(filter, planes);
            normalize(planes[0], filterOutput, 0, 1, CV_MINMAX);

            // ***

            // display image in window

            imshow(originalName, imgGray);
            imshow(spectrumMagName, mag);
            imshow(lowPassName, imgOutput);
            imshow(filterName, filterOutput);

            // start event processing loop (very important,in fact essential for GUI)
            // 40 ms roughly equates to 1000ms/25fps = 4ms per frame

            key = waitKey(EVENT_LOOP_DELAY);

            if (key == 'x')
            {

                // if user presses "x" then exit

                std::cout << "Keyboard exit requested : exiting now - bye!"
                          << std::endl;
                keepProcessing = false;
            }
        }

        // the camera will be deinitialized automatically in VideoCapture destructor

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}

double deg2rad( double a )
{
    return (M_PI*a)/180;
}

void create_a_model()
{
    cv::Mat img(360, 360, cv::DataType<unsigned char>::type);
    for ( size_t i = 0; i < img.cols; i+=1 )
    {
        for ( size_t j = 0; j < img.rows; j+=1 )
        {
            img.at<unsigned char>(i,j) = 127 + 127 * sin(deg2rad(20*j));
        }

    }
    imwrite("sin.bmp", img);
}