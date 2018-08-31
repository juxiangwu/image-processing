#include "particleFilterTracking.h"
#include "getInitialRect.h"
#include <iostream>
using namespace std;
int main (int argc, char** argv)
{
    //Mat img=imread("img2.jpeg");
    Mat img;
    VideoCapture video(0);
    while(1)
    {
        video>>img;
        imshow("GetInitialRect",img);
        char c= waitKey(100);
        if(c=='i')
            break;
    }
    InitialRect intialRect=InitialRect();
    Rect toTrack= intialRect.getInitialRect(img);
    ParticleFilterTrackor trackor=ParticleFilterTrackor();
    trackor.Initialize(img,toTrack);
    float maxWeight=0;
    while(1)
    {
        video>>img;
        int t=trackor.ColorParticleTracking(img,toTrack, maxWeight);
        cout<<t<<"  "<<maxWeight<<endl;
        rectangle(img,toTrack,Scalar(10,10,200),5);
        imshow("img",img);
        char c= waitKey(100);
        if(c=='q')
            break;
    }

    return 0;
}
