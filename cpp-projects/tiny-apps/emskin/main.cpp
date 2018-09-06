#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int patchX = 25; int patchY = 25;

cv::Rect patchGridRect(int idX, int idY) {
  return cv::Rect(idX * patchX, idY * patchY,
                  patchX, patchY);
}

void addPatchGrid(const cv::Mat& img, std::vector<cv::Mat>& samples,
                  int idX, int idY)
{
  cv::Rect patch = patchGridRect(idX, idY);

  samples.push_back(cv::Mat(img, patch).clone());
}

int main( int argc, char** argv ) {

  if( argc != 3 ) {
    std::cerr << "Usage: " << argv[0] << " <InputImage> <TrainImage>" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat image;
  image = cv::imread( argv[1] );
  if(!image.data) {
    return EXIT_FAILURE;
  }
  cv::Mat trainImage;
  trainImage = cv::imread( argv[2] );
  if(!trainImage.data) {
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> skinImages;
  addPatchGrid(image, skinImages, 5, 6);
  addPatchGrid(image, skinImages, 6, 7);
  addPatchGrid(image, skinImages, 7, 8);

  // cv::rectangle(image, patchGridRect(12, 10), cv::Scalar(0, 0, 255));
  // cv::rectangle(image, patchGridRect(12, 11), cv::Scalar(0, 0, 255));
  // cv::rectangle(image, patchGridRect(12, 12), cv::Scalar(0, 0, 255));
  // cv::rectangle(image, patchGridRect(12, 13), cv::Scalar(0, 0, 255));
  // cv::rectangle(image, patchGridRect(12, 14), cv::Scalar(0, 0, 255));
  // cv::rectangle(image, patchGridRect(12, 15), cv::Scalar(0, 0, 255));

  cv::Mat CbCrImage;
  cv::cvtColor( image, CbCrImage, CV_BGR2YCrCb );
  CbCrImage.convertTo(CbCrImage, CV_64F);

  cv::Mat trainSamples = cv::Mat(patchX*patchY*skinImages.size(), 2, CV_64FC1);
  for(size_t i = 0; i < skinImages.size(); ++i) {
    cv::cvtColor( skinImages[i], skinImages[i], CV_BGR2YCrCb );
    skinImages[i].convertTo(skinImages[i], CV_64FC3);

    for( int y = 0; y < skinImages[0].rows ; y++ ) {
      for( int x = 0; x < skinImages[0].cols; x++ ) {
        cv::Vec3d color = skinImages[0].at<cv::Vec3d>(y, x);

        trainSamples.at<double>(y*x+i*patchX*patchY, 0) = color[1];
        trainSamples.at<double>(y*x+i*patchX*patchY, 1) = color[2];
      }
    }
  }

  cv::Mat skinSample(1, 2, CV_64F);
  cv::Mat skinMask(CbCrImage.size(), CV_64FC1);

  cv::Ptr<cv::ml::EM> emSkin = cv::ml::EM::create();
  cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainSamples,cv::ml::SampleTypes::ROW_SAMPLE,CV_64F);

  if(emSkin->train(td)) {
    for( int j = 0; j < CbCrImage.rows ; j++ ) {
      for( int i = 0; i < CbCrImage.cols; i++ ) {
        cv::Vec3d color = CbCrImage.at<cv::Vec3d>(j, i);

        skinSample.at<double>(0) = color[1];
        skinSample.at<double>(1) = color[2];

        cv::Vec2d prob = emSkin->predict(skinSample);
        skinMask.at<double>(j, i, 0) = prob[0];
      }
    }
  } else {
    std::cerr << "Failed training Skin EM" << std::endl;
  }

  skinMask.convertTo(skinMask, CV_32FC1);

  cv::normalize(skinMask, skinMask, 0., 255., cv::NORM_MINMAX);
  cv::imwrite("mask1.jpg", skinMask);

  cv::threshold(skinMask, skinMask, 230., 255., cv::THRESH_BINARY);

  int erosion_size = 3;
  cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
                                          cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ));

  cv::dilate( skinMask, skinMask, element );
  cv::erode( skinMask, skinMask, element );

  cv::imwrite("mask2.jpg", skinMask);

  return EXIT_SUCCESS;
}
