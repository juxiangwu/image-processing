/*************************************************
*@Author   Xiangwei Wang
*@Date     2015 May 20th
*@Function A class to select a rectangle region from a frame
*@Usage    1. Include the header file 
		   2. Define a object with defaut constructor funtion
		   3. Call the getInitialRect funtion.
**************************************************/
#include <opencv2/opencv.hpp>
using namespace cv;
class InitialRect
{
public:
	static Mat imgGlobal;
	static Rect selection;
	static bool selectObject;
	static Point origin;

	InitialRect()
	{
		//selectObject=false;
	}
	static void onMouse( int event, int x, int y, int, void* )
	{
	    if( selectObject )
	    {
	        selection.x = origin.x;
	        selection.y = origin.y;
	        selection.width = std::abs(x - origin.x);
	        selection.height = std::abs(y - origin.y);

	        selection &= Rect(0, 0, imgGlobal.cols, imgGlobal.rows);
	        rectangle(imgGlobal,selection,Scalar(255,0,0));
	        
	    }

	    switch( event )
	    {
	    case CV_EVENT_LBUTTONDOWN:
	        origin = Point(x,y);
	        circle(imgGlobal,origin,5,Scalar(100,100,0),-1);
	        selection = Rect(x,y,0,0);
	        selectObject = true;
	        break;
	    case CV_EVENT_LBUTTONUP:
	        selectObject = false;
	        break;
	    }
	    imshow("GetInitialRect",imgGlobal);
	    waitKey(10);
	}
	Rect getInitialRect(Mat img)
	{
		img.copyTo(imgGlobal);
		namedWindow( "GetInitialRect");
		setMouseCallback( "GetInitialRect", onMouse, 0 );
		imshow( "GetInitialRect",imgGlobal);
		waitKey();
		return selection;
	}

};
bool  InitialRect::selectObject =false;
Rect  InitialRect::selection    =Rect(0,0,0,0);
Point InitialRect::origin       =Point(0,0);
Mat   InitialRect::imgGlobal    ;