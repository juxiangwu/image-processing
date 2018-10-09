#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

using namespace std;

int main()
{
    int num_devices = cv::cuda::getCudaEnabledDeviceCount();

    if(num_devices <= 0)
    {
        std::cerr<<"There is no device."<<std::endl;
        return -1;
    }
    int enable_device_id = -1;
    for(int i=0;i<num_devices;i++)
    {
        cv::cuda::DeviceInfo dev_info(i);
        if(dev_info.isCompatible())
        {
            enable_device_id=i;
        }
    }
    if(enable_device_id < 0)
    {
        std::cerr<<"GPU module isn't built for GPU"<<std::endl;
        return -1;
    }
    cv::cuda::setDevice(enable_device_id);

    std::cout<<"GPU is ready, device ID is "<<num_devices<<"\n";

    return 0;
}
