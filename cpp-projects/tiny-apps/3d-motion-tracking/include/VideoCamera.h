#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include "ITransformation.h"
#include "IVideoSource.h"

class VideoCamera : public IVideoSource
{
public:
  VideoCamera(string wndTitle="");
  virtual Mat captureWith(ITransformation &t, bool bShow) final;
  virtual void captureRealtimeWith(ITransformation &t, function<void (Mat)> pipe) final;

protected:
  // Instance of the collaborating camera
  VideoCapture m_camera;
  // Window title (set to empty string if running UI-less)
  string m_wndTitle;
};