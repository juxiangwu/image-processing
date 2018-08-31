/**
 * Simple test suite
 * @author Tao PR (github.com/starcolon)
 */

#include <functional>
#include <queue>
#include "IdentityTransformation.h"
#include "FilterTransformation.h"
#include "VideoCamera.h"
//#include "CamShiftTracker.h"
//#include "SimpleFeaturePointTracker.h"

#include "TestHungarian.h"

using namespace std;

extern TestScenario testHungarian;

int main(int argc, char** argv)
{
  // All test scenarios we want to examine
  vector<TestScenario> tests = {
    testHungarian
  };

  cout << "[3d-motion-track]" << endl;
  for (auto test : tests)
  {
    test.runAll();
  }
  
}
