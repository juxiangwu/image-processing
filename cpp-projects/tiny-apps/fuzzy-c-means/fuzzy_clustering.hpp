#ifndef FUZZY_CLUSTERING
#define FUZZY_CLUSTERING
#include <iostream>
#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>

#include <stdlib.h>

typedef enum {
  kSoftCInitRandom = 0,
  kSoftCInitKmeansPP,
} SoftCInitType;

typedef enum {
  kSoftCDistL1 = 0,
  kSoftCDistL2,
  kSoftCDistHistInter,    // Not implemented
} SoftCDistType;

namespace SoftC {

  typedef float Value;
  typedef unsigned int Index;

  class Fuzzy {
    public:

      Fuzzy (
          const cv::Mat &rows,
          const int number_clusters,
          const float fuzziness,
          const float epsilon,
          const SoftCDistType dist_type,
          const SoftCInitType init_type
          )
        :

        fuzziness_ (fuzziness),
        epsilon_ (epsilon),
        dist_type_ (dist_type),
        init_type_ (init_type),
        number_clusters_ (number_clusters),
        number_points_ (rows.rows),
        dimension_ (rows.cols),
        rows_ (rows)
        {
          centroids_ = cv::Mat::zeros  (number_clusters_, dimension_, CV_32FC1);
          membership_
            = cv::Mat::zeros  (number_points_, number_clusters_, CV_32FC1);
          new_membership_
            = cv::Mat::zeros  (number_points_, number_clusters_, CV_32FC1);
          initEverything ();
        };

      void initEverything ();
      void initRandom ();
      void initKmeansPP ();

      void computeCentroids ();
      void computeCentroids2 ();
      bool updateMembership ();

      float calc_dist (
          const cv::Mat &point,
          const cv::Mat &center,
          const SoftCDistType dist_type);

      inline const bool can_stop ()
      {
        float t = cv::norm (membership_ - new_membership_);
        return t < epsilon_;
      }

      inline void clustering (const unsigned int num_iteration = 10000) {
        unsigned int iteration = 0;

        while (!updateMembership () && iteration++ < num_iteration) {
          computeCentroids2();
        }
      }

      inline const cv::Mat get_centroids_ () { return centroids_; }
      inline const cv::Mat get_membership_ () { return membership_; }
      inline const cv::Mat get_new_membership_ () { return new_membership_; }

    private:

      float fuzziness_;
      float epsilon_;

      int number_clusters_;
      int number_points_;
      int dimension_;

      SoftCDistType dist_type_;
      SoftCInitType init_type_;

      cv::Mat centroids_;
      cv::Mat membership_;
      cv::Mat new_membership_;
      cv::Mat rows_;
  };
};
#endif
