/************************************************************************************
*Title:facemorph
*Author:Yiming Xia
*Library File: Eigen  OpenCV Dlib
*Tips:
*1.You can run this in Visual Studio 2015 or newer.Library file should be configured before compiling
*  Dlib and Eigen is included in this package,OpenCV is not.
*2.You should change the address of the shape_predictor_68_face_landmarks.dat
*   which is in the facedetect function.(It is included in the package)
*3.You should also change the adress of the two images with the people whose mouth is open
*  to get the .avi successful.
*4.Release is faster than Debug.
*************************************************************************************/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include"Eigen/Dense"
using namespace Eigen;
using namespace dlib;
using namespace cv;
using namespace std;

std::vector<cv::Point2f> facedetect(string pname)                                             //生产特征点
{
    std::vector<cv::Point2f> pointlist;
    Mat imgt = imread(pname);
    try
    {
        frontal_face_detector detector = get_frontal_face_detector();

        shape_predictor sp;//定义个shape_predictor类的实例


        deserialize("E:\\Maths\\Digital Image Processing\\HW\\HW2\\test\\Project2\\shape_predictor_68_face_landmarks.dat") >> sp;//设置人脸识别的数据包，路径为文件所在路径

        image_window win, win_faces;
        {
            array2d<rgb_pixel> img;//注意变量类型 rgb_pixel 三通道彩色图像
            load_image(img,pname);
            std::vector<dlib::rectangle> dets = detector(img);//检测人脸，获得边界框

            std::vector<full_object_detection> shapes;//注意形状变量的类型，full_object_detection
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);//预测姿势，注意输入是两个，一个是图片，另一个是从该图片检测到的边界框
                for (int i = 0; i < 68; i++)
                {
                    pointlist.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
                }
                shapes.push_back(shape);
            }

            pointlist.push_back(cv::Point2f(0,0));
            pointlist.push_back(cv::Point2f(0, imgt.rows - 1));
            pointlist.push_back(cv::Point2f(imgt.cols - 1, 0));
            pointlist.push_back(cv::Point2f(imgt.cols - 1, imgt.rows - 1));
            pointlist.push_back(cv::Point2f(imgt.cols/2, 0));
            pointlist.push_back(cv::Point2f(0, imgt.rows/2));
            pointlist.push_back(cv::Point2f(imgt.cols - 1, imgt.rows/2));
            pointlist.push_back(cv::Point2f(imgt.cols/2, imgt.rows - 1));

            return pointlist;

        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

std::vector<Vec3i> delaunayTri(Mat& img, std::vector<Point2f> points)   //将三角关系转化为索引关系
{
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);
    std::vector<Vec3i> delaunayTri;
    Subdiv2D subdiv(rect);
    for (std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
    }
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    std::vector<Point> pt(3);
    for (int i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            int ji = 0;
            cv::Vec3i ind;
            for (int j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < points.size(); k++)
                {
                    if (abs(pt[j].x - points[k].x) < 0.1   &&  abs(pt[j].y - points[k].y) < 0.1)
                    {
                        ind(j) = k;
                        ji++;
                    }
                }
            }
            if (ji > 3)//重合点检查，如果出现问号，则此图无法作为输入
            {
                cout << "?" << endl;
            }
            if (ji == 3)
            {
                delaunayTri.push_back(ind);
            }
        }
    }

    return delaunayTri;
}

MatrixXd Wraptransform(VectorXd before, VectorXd after)                                         //求变换矩阵
{
    MatrixXd M = MatrixXd::Zero(6, 6);
    MatrixXd Out = MatrixXd::Zero(3, 3);

    for (int i = 0; i < 3; i++)
    {
        M(2 * i, 0) = before(2 * i);
        M(2 * i, 1) = before(2 * i + 1);
        M(2 * i, 2) = 1;
        M(2 * i + 1, 3) = before(2 * i);
        M(2 * i + 1, 4) = before(2 * i + 1);
        M(2 * i + 1, 5) = 1;
    }
    VectorXd solve = M.colPivHouseholderQr().solve(after);

    for (int i = 0; i < 6; i++)
    {
        Out(i / 3, i % 3) = solve(i);
    }
    Out(2, 2) = 1;
    return Out;
}

Mat facemorph(float alpha, std::vector<cv::Point2f> input_pointlist, std::vector<cv::Point2f> output_pointlist, Mat img1, Mat img2, std::vector<Vec3i> trianglelist)
{

    int width = img1.cols;                   //图像的水平边
    int height = img1.rows;                  //图像的竖直边

    std::vector<cv::Point2f> morph_pointlist;
    for (int i = 0; i < 76; i++)
    {
        cv::Point2f morphpoint = (1 - alpha)*input_pointlist[i] + alpha*output_pointlist[i];
        morph_pointlist.push_back(morphpoint);
    }
    Mat record = img1.clone();
    Mat show = img1.clone();
    cv::RNG rng(time(0));
    for (int i = 0; i < trianglelist.size(); i++)
    {

        std::vector<Point> fill;
        fill.push_back(morph_pointlist[trianglelist[i](0)]);
        fill.push_back(morph_pointlist[trianglelist[i](1)]);
        fill.push_back(morph_pointlist[trianglelist[i](2)]);
        fillConvexPoly(record, fill, i);
        fillConvexPoly(show, fill, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    std::vector<MatrixXd> before_transferlist, after_transferlist;
    for (int i = 0; i < trianglelist.size(); i++)
    {
        MatrixXd before_transfer, after_transfer;
        VectorXd before = VectorXd::Zero(6);
        VectorXd after = VectorXd::Zero(6);
        VectorXd mid = VectorXd::Zero(6);
        for (int j = 0; j < 3; j++)
        {
            before(2 * j) = input_pointlist[trianglelist[i](j)].x;
            before(2 * j + 1) = input_pointlist[trianglelist[i](j)].y;
            after(2 * j) = output_pointlist[trianglelist[i](j)].x;
            after(2 * j + 1) = output_pointlist[trianglelist[i](j)].y;
            mid(2 * j) = morph_pointlist[trianglelist[i](j)].x;
            mid(2 * j + 1) = morph_pointlist[trianglelist[i](j)].y;
        }
        before_transfer = Wraptransform(mid, before);
        after_transfer = Wraptransform(mid, after);
        before_transferlist.push_back(before_transfer);
        after_transferlist.push_back(after_transfer);
    }

    Mat morphimage = Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i<height; i++)
    {
        for (int j = 0; j<width; j++)            //j是x，i是y
        {
            int num;
            num = record.at<Vec3b>(i, j)[0];

            Vector3d coordinate(j, i, 1);
            Vector3d beforecoordinate = before_transferlist[num] * coordinate;
            Vector3d aftercoordinate = after_transferlist[num] * coordinate;
            int beforex = int(beforecoordinate(0));
            int beforey = int(beforecoordinate(1));
            int afterx = int(aftercoordinate(0));
            int aftery = int(aftercoordinate(1));
            morphimage.at<Vec3b>(i, j) = (1 - alpha)*img1.at<Vec3b>(beforey, beforex) + alpha*img2.at<Vec3b>(aftery, afterx);
        }
    }

    return morphimage;
}


int main()
{
    //设置输入图像的路径
    string adress1 = "E:\\Maths\\Digital Image Processing\\Practice\\xia1.jpg";
    string adress2 = "E:\\Maths\\Digital Image Processing\\Practice\\wuyanzu1.jpg";

    Scalar points_color(0, 0, 255);

    std::vector<Vec6f> tri1;
    std::vector<cv::Point2f> input_pointlist = facedetect(adress1);                      //生成特征点
    Mat img1 = imread(adress1);
    std::vector<Vec6f> tri2;
    std::vector<cv::Point2f> output_pointlist = facedetect(adress2);                      //生成特征点
    Mat img2 = imread(adress2);
    Mat test = img1.clone();
    std::vector<Vec3i> trianglelist = delaunayTri(img1, input_pointlist);                  //生成三角索引关系，两张图共用此索引关系


    std::vector<Mat> pic;
    pic.push_back(img1);
    for (double alpha = 0.01; alpha <0.99; alpha = alpha + 0.01)
    {
        Mat morphimage = facemorph(alpha, input_pointlist, output_pointlist, img1, img2, trianglelist);
        pic.push_back(morphimage);
    }
    pic.push_back(img2);
    cout << "Finish!" << endl;
    VideoWriter output_src("facemorph.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24, pic[0].size(), 1);   //输出.avi文件
    for (auto c : pic)
    {
        output_src << c;
    }
}
