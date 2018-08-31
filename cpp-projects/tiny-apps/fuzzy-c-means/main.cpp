
#include "fuzzy_clustering.hpp"

#include <opencv/cxcore.h>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <time.h>
#include <sstream>

//#include <glog/logging.h>

#include <opencv2/core/core.hpp>

int main (int argc, char **argv) {
//  google::InitGoogleLogging (argv[0]);
//  google::InstallFailureSignalHandler ();

//  static unsigned int number_points = 17;
//  static unsigned int dimension_point = 4;
  static unsigned int number_clusters = 2;
//
//  srand((unsigned)time(0));
//
//  cv::Mat dataset (number_points, dimension_point, CV_32FC1);
//  for (int j = 0; j < dataset.rows; ++j) {
//    for (int  i = 0; i < dataset.cols; ++i) {
//      dataset.at<float> (j, i) = (float) rand() / (float) RAND_MAX;
//    }
//  }

  cv::Mat dataset = (cv::Mat_<float> (4, 2)
    <<
    0, 0,
    5, 4,
//    3.3, 2,
//    15, 10,
//    7.5, 6,
//    30, 40.3,
//    50, 60,
//    70, 80,
//    90, 100,
//    100, 200.252,
//    150, 151,
//    200, 205,
    100, 150,
    200, 102);

  float fuzziness = 2.0;    // initial: 1.1
  float epsilon = 0.01;
  SoftCDistType dist_type = kSoftCDistL2;
  SoftCInitType init_type = kSoftCInitKmeansPP;

  SoftC::Fuzzy f (dataset, number_clusters, fuzziness, epsilon, dist_type, init_type);

  // Note: This does not mean iteration initization.
  unsigned int num_iterations = 100;
  f.clustering (num_iterations);

  std::cout << "### Results ### " << std::endl;

  cv::Mat centroids = f.get_centroids_ ();
  std::cout << centroids << std::endl;

  cv::Mat memberships = f.get_membership_ ();
  std::cout << memberships << std::endl;
}
