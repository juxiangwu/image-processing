#include "Particle.h"

Particle::Particle(Point2f &p)
{
  int N = 0;
  this->p0 = Point2f(0,0);
  this->absenceLength = 0;
  this->kf = KalmanFilter(
    DIMENSION_OF_STATE, 
    DIMENSION_OF_MEASUREMENT,
    DIMENSION_OF_CONTROLVECTOR,
    CV_32F);

  // Observation Linear Model:
  // z[k] = H[k]*x[k] + v[k]

  // Noise {v}
  this->noise = Mat(DIMENSION_OF_MEASUREMENT, 1, CV_32F);

  // State {x} : [x, y, dx/dt, dy/dt]
  this->state = Mat(DIMENSION_OF_STATE, DIMENSION_OF_STATE, CV_32F);
  setIdentity(this->state);
  this->kf.statePre.at<float>(0) = p.x;
  this->kf.statePre.at<float>(1) = p.y;
  this->kf.statePre.at<float>(2) = 0; // Nil speed x
  this->kf.statePre.at<float>(3) = 0; // Nil speed y

  // Measurement {z}
  this->measure = Mat(DIMENSION_OF_MEASUREMENT, 1, CV_32F);

  // Measurement emission matrix {H} : x -> z
  this->kf.measurementMatrix = Mat::zeros(
    DIMENSION_OF_MEASUREMENT,
    DIMENSION_OF_STATE,
    CV_32F);
  N = DIMENSION_OF_STATE;
  this->kf.measurementMatrix.at<float>(0 + N*0) = 1; // x(measure) = x(state)
  this->kf.measurementMatrix.at<float>(1 + N*1) = 1; // y(measure) = y(state)

  // Transition Matrix A : A*x[k] = x[k+1] .... (kinetic system)
  setIdentity(this->kf.transitionMatrix);
  float dt = 0.1;
  N = DIMENSION_OF_MEASUREMENT;
  this->kf.transitionMatrix.at<float>(2 + N*0) = dt; // x' = x + v[x]*dt
  this->kf.transitionMatrix.at<float>(3 + N*1) = dt; // y' = y + v[y]*dt

  N = DIMENSION_OF_STATE;
  this->kf.processNoiseCov = Mat::zeros(DIMENSION_OF_STATE, DIMENSION_OF_STATE, CV_32F);
  this->kf.processNoiseCov.at<float>(0 + N*0) = 1e-5; // R_x
  this->kf.processNoiseCov.at<float>(1 + N*1) = 1e-5; // R_y
  this->kf.processNoiseCov.at<float>(2 + N*2) = 0.1; // R_vx
  this->kf.processNoiseCov.at<float>(3 + N*3) = 0.1; // R_vy

  // Measurement Noise covariance : v[k] ~ N(0,R[k]) .... where E[v.vT] = R
  setIdentity(this->kf.measurementNoiseCov, Scalar::all(1e-2));
}

Point2f Particle::update(Point2f &p)
{
  // Predict the state
  Point2f p_ = this->predict();

  // Update the measurement
  int N = DIMENSION_OF_MEASUREMENT;
  Mat m = (Mat_<float>(N,1) << p.x, p.y);
  this->kf.correct(m);

  return p_;
}

Point2f Particle::predict()
{
  Mat matp = this->kf.predict();
  this->p_ = Point2f(matp.at<float>(0), matp.at<float>(1));
  return this->p_;
}


Point2f Particle::setAbsence()
{
  // Predict the state
  Point2f p_ = this->predict();

  this->absenceLength ++;
  return p_;
}



