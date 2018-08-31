#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

bool OVERLAY_ENABLED = true;
bool VALID_H = false;

void readme();
void SURFAlgorithm( Mat& scene_img, Ptr<Feature2D>& surf,
            vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
            Mat& descriptors1, Mat& descriptors2, Mat& H, bool& Hrel);
void drawBox(std::vector<Point2f> &scene_corners, Mat &img_show);
void getCameraData(std::string filename, Mat& cameraMatrix2, Mat& distCoeffs2);
void drawAxis( std::vector<Point2f> &imagePoints, Mat &img_show);


/* @function main */
int main(int argc, char* argv[])
{
//    if( argc < 2 )
//    { readme(); return -1; }

    //-- Read objectImage ( the object to be "detected" )
    Mat objectImage = imread("d:/develop/dl/projects/resources/images/ball.jpg");
    Mat gray_object;
    cvtColor(objectImage, gray_object, COLOR_BGR2GRAY);

    //-- Get the corners from the objectImage ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( objectImage.cols, 0 );
    obj_corners[2] = cvPoint( objectImage.cols, objectImage.rows ); obj_corners[3] = cvPoint( 0, objectImage.rows );
    vector<Point2f> scene_corners(4);
    vector<Point3f> objectPoints(4);
    vector<Point3f> axis(4);
    vector<Point2f> imagePoints(4);

    //-- Create 3D corners from the objectImage ( the object to be "detected" )
    for ( int i = 0; i < (int)obj_corners.size(); i++ )
    {
        objectPoints[i] = Point3f(obj_corners[i]);
    }

    //-- 3D object to render
    axis[0] = Point3f(0.0, 0.0, 0.0);
    axis[1] = Point3f(50.0, 0.0, 0.0);
    axis[2] = Point3f(0.0, 50.0, 0.0);
    axis[3] = Point3f(0.0, 0.0, -50.0);

    VideoCapture cap(0); // open the video file for reading
    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return -1;
    }

    cap.set(CAP_PROP_POS_MSEC, 300); //start the video at 300ms
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;
    cout << "Width : "  << cap.get(CAP_PROP_FRAME_WIDTH) << endl; // Width of the frames in the video stream.
    cout << "Height : "  << cap.get(CAP_PROP_FRAME_HEIGHT) << endl; // Height of the frames in the video stream.
    cout << "Frame per seconds : " << fps << endl;
    imshow( "Target Object", objectImage );
    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    // Load camera calibration data
    Mat cameraMatrix, distCoeffs;
    string filename = "d:/develop/dl/projects/resources/models/opencv/slam-out_camera_data.xml";
    getCameraData(filename, cameraMatrix, distCoeffs);
    cout << "camera matrix: " << cameraMatrix << endl
         << "distortion coeffs: " << distCoeffs << endl;

    // Homography matrix, translation and rotation vectors
    Mat frame, H;
    Mat rvec(3,1,cv::DataType<double>::type);
    Mat tvec(3,1,cv::DataType<double>::type);

    //-- Detect objectImage keypoints and extract descriptors using FastFeatureDetector
    Ptr<Feature2D> surf = SURF::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    surf->detectAndCompute(gray_object, Mat(), keypoints1, descriptors1);

    while(1)
    {
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        // Compute Homography matrix H
        SURFAlgorithm( frame, surf, keypoints1, keypoints2,
            descriptors1, descriptors2, H, VALID_H);

        // Project object corners in the scene prespective
        perspectiveTransform( obj_corners, scene_corners, H);

        // Compute the perspective projection of the 3D object to be rendered in the scene
        solvePnP(objectPoints, scene_corners, cameraMatrix, distCoeffs, rvec, tvec);

        // Project 3D object in the scene
        projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

        // Display output
        if (VALID_H)
        {
            drawBox(scene_corners, frame);
            drawAxis(imagePoints, frame);
        }
        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }
    return 0;
}

/* @function readme */
void readme()
{ std::cout << " Usage: ./rtTracker <target_img> " << std::endl; }


void drawBox(vector<Point2f> &scene_corners, Mat &img_show)
// Draw the contour of the detected object
{
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_show, scene_corners[0], scene_corners[1], Scalar( 255, 255, 0), 4 );
    line( img_show, scene_corners[1], scene_corners[2], Scalar( 255, 255, 0), 4 );
    line( img_show, scene_corners[2], scene_corners[3], Scalar( 255, 255, 0), 4 );
    line( img_show, scene_corners[3], scene_corners[0], Scalar( 255, 255, 0), 4 );
}

void drawAxis(vector<Point2f> &axis, Mat &img_show)
// Render 3D object
{
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_show, axis[0], axis[1], Scalar( 255, 0, 0), 5 );
    line( img_show, axis[0], axis[2], Scalar( 0, 255, 0), 5 );
    line( img_show, axis[0], axis[3], Scalar( 0, 0, 255), 5 );
}

void getCameraData(std::string filename, Mat& cameraMatrix2, Mat& distCoeffs2)
{
    FileStorage fs2(filename, FileStorage::READ);

    fs2["camera_matrix"] >> cameraMatrix2;
    fs2["distortion_coefficients"] >> distCoeffs2;

    fs2.release();
}

void SURFAlgorithm( Mat& scene_img, Ptr<Feature2D>& surf,
            vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
            Mat& descriptors1, Mat& descriptors2, Mat& H, bool& VALID_H)
{
    Mat gray_scene, H1, outlier_mask;
    cvtColor( scene_img, gray_scene, COLOR_BGR2GRAY);
    if( !gray_scene.data )
    {
        std::cout<< " --(!) Error reading scene image " << std::endl;
    }

    //-- Step 1: Detect the keypoints and extract descriptors using FastFeatureDetector
    surf->detectAndCompute(scene_img, Mat(), keypoints2, descriptors2);

    //-- Step 2: Find the closest matches between descriptors from the first image to the second
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
    //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ matches[i].trainIdx ].pt );
    }
    H = findHomography( obj, scene, RANSAC, 3, outlier_mask );
    if (sum( outlier_mask )[0] > 40)
        {VALID_H = true;}
    else
        {VALID_H = false;};
}
