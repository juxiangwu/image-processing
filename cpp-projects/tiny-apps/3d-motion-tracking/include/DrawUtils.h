#ifndef DRAWUTILS
#define DRAWUTILS

#include <vector>
#include <math.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cvaux.hpp>

using namespace cv;
using namespace std;

class DrawUtils
{
public:

  inline static void drawSpot(Mat& canvas, Point2d p, Scalar color)
  {
    circle(canvas, p, 3, color, CV_FILLED, CV_AA);
  }
  
  /**
   * Draw marks spots on the canvas
   * @param {Mat} canvas matrix
   * @param {Mat} matrix containing marks to draw
   * @param {Scalar} color to paint
   */
  inline static void drawMarks(Mat& canvas, Mat& marks, Scalar color)
  {
    for (int j=0; j<canvas.rows; j++)
      for (int i=0; i<canvas.cols; i++)
      {
        Scalar cmark = marks.at<float>(j, i);
        if (cmark.val[0]>0)
        {
          circle(canvas, Point(i, j), 1, color, 1, CV_AA);
        }
      }
  }

  /**
   * Draw mark spots on the canvas
   * @param {Mat} canvas matrix
   * @param {vector<Point2f>} list of markers to draw
   * @param {Scalar} color to paint
   */
  inline static void drawMarks(Mat& canvas, vector<Point2f> marks, Scalar color)
  {
    for (Point2f p : marks)
    {
      circle(canvas, p, 3, color, CV_FILLED, CV_AA);
    }
  }

private:
};

#endif