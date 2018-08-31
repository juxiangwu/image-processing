#include "fuzzy_clustering.hpp"
#include <boost/foreach.hpp>
#include <iostream>
#include <time.h>
namespace SoftC {

  void Fuzzy::initRandom () {
    srand ((unsigned int) time (NULL));
    float normalization_factor;
    for (int j = 0 ; j < number_points_; j++) {
      normalization_factor = 0.0;
      for (int i = 0; i < number_clusters_; i++)
        normalization_factor +=
          membership_.at<float> (j, i) = (rand () / (RAND_MAX + 0.0));
      // Normalization
      for (int i = 0; i < number_clusters_; i++)
        membership_.at<float> (j, i) /= normalization_factor;
    }
    computeCentroids();
  }

  void Fuzzy::initKmeansPP () {
    srand ((unsigned int) time (NULL));
    std::vector<int> center_indexes (0);
    std::vector<bool> already_selected_indexes (number_points_, false);

    // Select first cluster centroids by random.
    int first_index = rand () % number_points_;
    center_indexes.push_back (first_index);
    already_selected_indexes[first_index] = true;

    while (center_indexes.size () < number_clusters_) {
      std::vector<float> nearest_distances (number_points_, 0.0);
      for (int p = 0; p < number_points_; ++p) {
        if (already_selected_indexes[p])
          continue;

        cv::Mat point = rows_.row (p);
        std::vector<float> distances_from_centers (0);

        // Calculate distances from all centroids.
        for (int c = 0; c < center_indexes.size (); ++c) {
          int center_index = center_indexes[c];
          cv::Mat center = rows_.row (center_index);
          float dist = calc_dist (point, center, kSoftCDistL2);
          distances_from_centers.push_back (dist);
        }

        // Find nearest centroid.
        int nearest_center_index = center_indexes[0];
        float min = distances_from_centers[0];
        for (int c = 1; c < distances_from_centers.size (); ++c) {
          float dist = distances_from_centers[c];
          if (dist < min) {
            min = dist;
            nearest_center_index = center_indexes[c];
          }
        }
        nearest_distances[p] = min;
      }

      assert (nearest_distances.size () == number_points_);

      float max = nearest_distances[0];
      float max_index = 0;
      for (int p = 1; p < nearest_distances.size (); ++p) {
        float dist = nearest_distances[p];
        if (dist > max) {
          max = dist;
          max_index = p;
        }
      }
      center_indexes.push_back (max_index);
      already_selected_indexes[max_index] = true;
    }

    for (int j = 0; j < center_indexes.size (); ++j) {
      for (int d = 0; d < dimension_; ++d) {
        // FIXME: Avoid zero devide.
        centroids_.at<float> (j, d) = rows_.at<float> (center_indexes[j], d) + 0.001;
      }
    }

    updateMembership ();
    computeCentroids2();
  }

  void Fuzzy::initEverything () {
    switch (init_type_) {
      case kSoftCInitRandom:
        initRandom ();
        break;
      case kSoftCInitKmeansPP:
        initKmeansPP ();
        break;
      default:
        break;
    }
  }

  void Fuzzy::computeCentroids(){
    // Update centroids
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        for (int f = 0; f < dimension_; f++)
          centroids_.at<float> (j, f) += membership_.at<float> (i, j) * rows_.at<float> (i, f);
    //    *p_centroids_ = prod (*p_membership_,        rows_);
    //   n_clusters   =      n_clusters          rows.size1()
    // X [rows.size2()=      X [rows.size1()=    X [rows.size2=
    //    =size_of_a_point_]    =number_points_]    size_of_a_point]
    std::vector<float> sum_uk (number_clusters_, 0);
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        sum_uk[j] += membership_.at<float> (j, i);
    // Normalization
    for (int j = 0; j < number_clusters_; j++)
      for (int f = 0 ; f < dimension_; f++)
        centroids_.at<float> (j, f) /= sum_uk[j];
  }

  void Fuzzy::computeCentroids2 (){
    cv::Mat u_ji_m  = cv::Mat::zeros (number_points_, number_clusters_, CV_32FC1);

    //　Initialization
    for (int j = 0; j < number_clusters_; j++)
      for (int f = 0; f < dimension_; f++)
        centroids_.at<float> (j, f) = 0.0;
    // weight ** fuzziness
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        u_ji_m.at<float> (i, j) = pow ( membership_.at<float> (i, j), fuzziness_);
    // Update centroids
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        for (int f = 0; f < dimension_; f++)
          centroids_.at<float> (j, f) += u_ji_m.at<float> (i, j) * rows_.at<float> (i, f);

    // Normalization
    float normalization;
    for (int j = 0; j < number_clusters_; j++){
      normalization = 0.0;
      for (int i = 0 ; i < number_points_; i++)
        normalization += u_ji_m.at<float> (i, j);
      for (int f = 0; f < dimension_; f++)
        centroids_.at<float> (j, f) /= normalization;
    }
  }

  float Fuzzy::calc_dist (
      const cv::Mat &point,
      const cv::Mat &center,
      const SoftCDistType dist_type
      )
  {
    float f_dist = 0.f;
    int dimension = point.cols;

    switch (dist_type) {
      case kSoftCDistL1:
        {
          // L1, Manhattan
          for (int d = 0; d < dimension; d++) {
            f_dist += fabs (point.at<float> (0,d) - center.at<float> (0,d));
          }
        }
        break;
      case kSoftCDistL2:
        {
          // L2, Euclid
          for (int d = 0; d < dimension; d++) {
            float t = point.at<float> (0,d) - center.at<float> (0,d);
            f_dist += t * t;
          }
        }
        break;
      case kSoftCDistHistInter:   // TODO
        {
          // HIstogram intersection
          float sum_p = 0.f;
          for (int d = 0; d < dimension; d++) {
            float p = point.at<float> (0,d);
            float c = center.at<float> (0,d);
            float min = p < c ? p : c;
            f_dist += min;
//            f_dist += ((p + c - fabs (p - c)) / 2);
            sum_p += p;
          }
          f_dist /= sum_p;
        }
        break;
      default:
        std::cout << "Error while calculating distance for clustering" << std::endl;
        break;
    }
    return f_dist;
  }

  bool Fuzzy::updateMembership () {
    cv::Mat matrix_norm_one_xi_minus_cj = cv::Mat::zeros (number_clusters_, number_points_, CV_32FC1);
    // Initialization
    for (unsigned int i = 0 ; i < number_points_; i++)
      for (unsigned int j = 0; j < number_clusters_; j++)
        matrix_norm_one_xi_minus_cj.at<float> (j, i) = 0.0;

    for (unsigned int i = 0 ; i < number_points_; i++) {
      // Calculate distances from each cluter.
      cv::Mat point = rows_.row (i);
      for (unsigned int j = 0; j < number_clusters_; j++) {
        cv::Mat center = centroids_.row (j);
        matrix_norm_one_xi_minus_cj.at<float> (j, i)
          = calc_dist (point, center, dist_type_);
      }
    }

    float coeff;
    for (unsigned int i = 0 ; i < number_points_; i++) {
      for (unsigned int j = 0; j < number_clusters_; j++) {
        coeff = 0.0;
        for (unsigned int k = 0; k < number_clusters_; k++) {

//        if (matrix_norm_one_xi_minus_cj.at<float> (j, i) == 0) {
//          coeff += pow (0, 2.0 / (fuzziness_ - 1.0));
//        } else if (matrix_norm_one_xi_minus_cj.at<float> (k, i) == 0) {
//          coeff += pow (1000000.0, 2.0 / (fuzziness_ - 1.0));
//        } else {
          coeff +=
            pow ( (matrix_norm_one_xi_minus_cj.at<float> (j, i) /
                  matrix_norm_one_xi_minus_cj.at<float> (k, i)) ,
                2.0 / (fuzziness_ - 1.0) );
        }

//        if (coeff == 0) {
//          new_membership_.at<float> (i, j) = 1.0;
//        } else {
          new_membership_.at<float> (i, j) = 1.0 / coeff;
//        }
      }
    }

    if (!can_stop() ){
      membership_ = new_membership_.clone ();
      return false;
    }
    return true;
  }
};
