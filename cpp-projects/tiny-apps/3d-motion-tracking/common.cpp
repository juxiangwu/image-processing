#include "common.h"

TestCase::TestCase(string title, function<bool()> func)
{
  this->title = title;
  this->func = func;
}

bool TestCase::pass() const
{
  return this->func();
}

TestScenario::TestScenario(string title)
{
  this->title = title;
}

TestScenario::~TestScenario()
{ 
  queue<TestCase>().swap(this->cases); 
}


void TestScenario::addCase(const TestCase &c)
{
  this->cases.push(c);
}

bool TestScenario::runAll()
{
  bool br = true;
  cout << GREEN << "[Test] " << this->title << endl << RESET;
  while (!this->cases.empty())
  {
    auto nextCase = this->cases.front();
    this->cases.pop();

    cout << CYAN << "...case# " << nextCase.getTitle() << endl << RESET;
    auto rc = nextCase.pass();
    if (!rc)
      cout << RED << "...[FAILED]" << endl << RESET;
    else
      cout << GREEN << "...[PASSED]" << endl << RESET;
    br &= rc;
  }

  return br;
}

TestScenario& operator>>(TestScenario& t, const TestCase &c)
{
  t.addCase(c);
  return t;
}

bool SkippedTestCase::pass() const
{
  cout << CYAN << "[SKIPPED: " << this->title << "]" << endl;
  return true;
}

bool isEql(Mat a, Mat b)
{
  assert(a.cols==b.cols && a.rows==b.rows);
  for (int j=0; j<a.rows; j++)
    for (int i=0; i<a.cols; i++)
    {
      if (a.at<float>(j,i) != b.at<float>(j,i))
      {
        cout << RED;
        printf("Inequal elem at (%d, %d)\n", j, i);
        cout << RESET;
        return false;
      }
    }
  return true;
}

bool isEql(set<int> a, set<int> b)
{
  return (a==b);
}

bool isEql(int a, int b)
{
  return (a==b);
}