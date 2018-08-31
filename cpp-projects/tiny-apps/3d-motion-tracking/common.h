#ifndef TEST_COMMON
#define TEST_COMMON

#include <set>
#include <queue>
#include <string>
#include <functional>
#include <iostream>
#include <assert.h>
#include "coloriser.h"
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.hpp>

using namespace std;
using namespace cv;

class TestCase
{
public:
  TestCase(string title, function<bool()> func);

  virtual bool pass() const;
  inline string getTitle() const { return this->title; }

protected:
  string title;
  function<bool()> func;

};


class SkippedTestCase : public TestCase 
{
public:
  using TestCase::TestCase;
  bool pass() const override;
};


class TestScenario
{
public:
  TestScenario(string title);
  ~TestScenario();

  void addCase(const TestCase &c);
  bool runAll();

  friend TestScenario& operator>>(TestScenario& t, const TestCase &c);

protected:
  string title;
  queue<TestCase> cases;

private:

};


TestScenario& operator>>(TestScenario& t, const TestCase &c);

bool isEql(Mat a, Mat b);
bool isEql(set<int> a, set<int> b);
bool isEql(int a, int b);

#endif