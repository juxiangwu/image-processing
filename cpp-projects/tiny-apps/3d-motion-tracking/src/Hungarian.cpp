#include "Hungarian.h"

bool CompareProfile::operator()(Profile &a, Profile &b )
{
  return get<1>(a) < get<1>(b);
}

bool CompareZeroInCol::operator()(ZeroInCol &a, ZeroInCol &b)
{
  return get<1>(a) > get<1>(b);
}

Hungarian::Hungarian(const Mat& cost, bool debug=false)
{
  this->debug = debug;
  this->costM = cost.clone();
}

vector<tuple<int,int>> Hungarian::optimiseMinima() const
{
  int nRows = this->costM.rows;
  int nCols = this->costM.cols;

  Mat cost = this->costM.clone();

  // Row-wise minima subtraction
  for (int n=0; n<nRows; n++)
  {
    float mini = Hungarian::minOfRow(n,cost);
    Mat minVec  = Mat(1, nCols, CV_32FC1, Scalar(mini));
    cost.row(n) = cost.row(n) - minVec;
  }

  if (this->debug)
  {
    cout << "...Row subtracted" << endl;
    cout << cost << endl << endl;
  }

  // Col-wise minima subtraction
  for (int n=0; n<nCols; n++)
  {
    float mini = Hungarian::minOfCol(n,cost);
    Mat minVec = Mat(nRows, 1, CV_32FC1, Scalar(mini));
    cost.col(n) = cost.col(n) - minVec;
  }

  if (this->debug)
  {
    cout << "...Col subtracted" << endl;
    cout << cost << endl << endl;
  }

  // Cover all zeroes in the matrix 
  // The minimum number of lines to cover must equal to the dimension
  int iter = 0;
  while (iter<MAX_ITER)
  {
    tuple<set<int>, set<int>> zeroes = Hungarian::coverZeroes(cost, this->debug);
    set<int> zeroRows = get<0>(zeroes);
    set<int> zeroCols = get<1>(zeroes);
    if (debug)
    {
      cout << cost << endl << endl;
      cout << "...Cover zeroes : expected {" << nRows 
        << "} got {" << zeroRows.size() + zeroCols.size() << "}" << endl;
    }
    if (zeroRows.size() + zeroCols.size() >= nRows)
      break;

    if (this->debug)
      cout << "...Creating additional zeroes" << endl;
    
    Hungarian::createAdditionalZeros(cost, zeroes, this->debug);
    iter++;
  }

  if (debug)
  {
    cout << "...[Original Mat]" << endl;
    cout << this->costM << endl << endl;
    cout << "...[Minima Mat]" << endl;
    cout << cost << endl << endl;
  }

  // Locate zeroes in reduced cost matrix
  priority_queue<ZeroInCol, vector<ZeroInCol>, CompareZeroInCol> minimaQ; // Least zero in a column comes first
  unordered_map<int,vector<int>> zeroesInCol;
  unordered_map<int,vector<int>> zeroesInRow;
  for (int i=0; i<nCols; i++)
  {
    int nZeroes = 0;
    for (int j=0; j<nRows; j++)
    {
      if (cost.at<float>(j,i) < 1E-4) // Zero?
      {
        nZeroes++;
        
        // Record found zeroes
        if (zeroesInCol.find(i)==zeroesInCol.end())
        {
          vector<int> v = {j};
          zeroesInCol.insert(make_pair(i,v));
        }
        else
          zeroesInCol[i].push_back(j);

        if (zeroesInRow.find(j)==zeroesInRow.end())
        {
          vector<int> v = {i};
          zeroesInRow.insert(make_pair(j,v));
        }
        else
          zeroesInRow[j].push_back(i);
      }
    }
    minimaQ.push(make_tuple(i,nZeroes));
  }

  // Select the optimal solution
  set<int> pickedCol;
  unordered_map<int,int> mapRowToCol;
  vector<tuple<int,int>> minima;
  if (debug)
  {
    cout << "...[Locating optimal solution]" << endl;
  }

  while (!minimaQ.empty())
  {
    // Take next column with least zeroes (least choices)
    auto next = minimaQ.top();
    int i = get<0>(next);
    if (debug) cout << "...take col #" << i << endl;

    // Take a zero row with "least" neighbor zeroes
    priority_queue<ZeroInCol,vector<ZeroInCol>,CompareZeroInCol> qRow;
    for (int j=0; j<nRows; j++)
    {
      // Skip the occupied row
      if (mapRowToCol.find(j) != mapRowToCol.end())
        continue;
      // Count number of unoccupied zeroes along the row [j]
      int numNeighborZero = 0;
      for (int ii : zeroesInRow[j])
      {
        if (pickedCol.find(ii) == pickedCol.end())
          numNeighborZero++;
      }

      qRow.push(make_tuple(j,numNeighborZero));
    }

    int j = get<0>(qRow.top());

    zeroesInCol[i].pop_back();
    pickedCol.insert(i);

    if (debug) printf("......match row # %d\n",j);

    // Memorise the selected optimum coordinate (j,i)
    mapRowToCol.insert(make_pair(j,i));
    minima.push_back(make_tuple(j,i));

    if (!zeroesInCol[i].empty()) zeroesInCol[i].pop_back();
    if (!minimaQ.empty()) minimaQ.pop();
  }

  if (debug)
  {
    cout << "...[Minima at]" << endl;
    for (int j=0; j<nRows; j++)
    {
      int iMin = mapRowToCol[j];
      for (int i=0; i<nCols; i++)
      {
        if (i==iMin) cout << "[†] ";
        else cout << "[ ] ";
      }
      cout << endl;
    }
  }

  return minima;
}

float Hungarian::minOfRow(int i, const Mat& cost)
{
  float m = numeric_limits<float>::max();
  int nCols = cost.cols;
  for (int k=0; k<nCols; k++)
  {
    float v = cost.at<float>(i,k);
    m = v < m ? v : m;
  }
  return m;
}

float Hungarian::minOfCol(int i, const Mat& cost)
{
  float m = numeric_limits<float>::max();
  int nRows = cost.rows;
  for (int k=0; k<nRows; k++)
  {
    float v = cost.at<float>(k,i);
    m = v < m ? v : m;
  }
  return m;
}

void Hungarian::createAdditionalZeros(Mat& m, tuple<set<int>, set<int>> zeroes, bool debug=false)
{
  // Find minimum uncovered number in {m}
  float minVal = numeric_limits<float>::max();
  int nRows = m.rows;
  int nCols = m.cols;

  set<int> zeroRows = get<0>(zeroes);
  set<int> zeroCols = get<1>(zeroes);

  for (int j=0; j<nRows; j++)
    for (int i=0; i<nCols; i++)
    {
      if (zeroRows.find(j) != zeroRows.end() || 
        zeroCols.find(i) != zeroCols.end())
        continue;
      float u = m.at<float>(j,i);
      if (u<minVal) minVal = u;
    }

  if (debug)
  {
    cout << "Minimum uncovered : " << minVal << endl;
  }

  // Subtract all uncovered numbers with {min}
  // Add {min} to double-covered numbers
  for (int j=0; j<nRows; j++)
    for (int i=0; i<nCols; i++)
    {
      bool rowCovered = zeroRows.find(j) != zeroRows.end();
      bool colCovered = zeroCols.find(i) != zeroCols.end();

      // Covered twice
      if (rowCovered && colCovered)
        m.at<float>(j,i) += minVal;
      // Uncovered
      else if (!rowCovered && !colCovered)
        m.at<float>(j,i) -= minVal;
    }

  if (debug)
  {
    cout << "Additional Zeroes:" << endl;
    cout << m << endl << endl;
  }

}

tuple<set<int>, set<int>> Hungarian::coverZeroes(Mat& m, bool debug=false)
{
  // Initialisation
  set<int> coverRowLines;
  set<int> coverColLines;
  unordered_map<int, int> profile;
  vector<Point> uncoveredZeroes;

  int numRows = m.rows;
  int numCols = m.cols;

  // Locate all zeroes
  for (int j=0; j<numRows; j++)
    for (int i=0; i<numCols; i++)
      if (m.at<float>(j,i) <= 1e-4) // Zero?
      {
        // Record the profiling lines
        if (profile.find(j+1) == profile.end())
          profile.insert(make_pair(j+1,1));
        else
          profile[j+1]++;

        if (profile.find(-i-1) == profile.end())
          profile.insert(make_pair(-i-1,1));
        else 
          profile[-i-1]++;

        // Record uncovered zeroes
        uncoveredZeroes.push_back(Point(i,j));
      }

  // if (debug)
  // {
  //   cout << "...[Initial uncovered zeroes]" << endl;
  //   for (auto z : uncoveredZeroes)
  //   {
  //     printf("...(%d, %d)\n", z.x, z.y);
  //   }
  // }

  // Convert profiles to priority queue
  priority_queue<Profile,vector<Profile>,CompareProfile> q0;
  priority_queue<Profile,vector<Profile>,CompareProfile> q;
  for (auto p : profile)
  {
    int k = get<0>(p);
    int mag = get<1>(p);
    q.push(make_tuple(k,mag));
    q0.push(make_tuple(k,mag));
  }

  if (debug)
  {
    cout << "...[Zero profiles]" << endl;
    while (!q0.empty())
    {
      auto p = q0.top();
      if (get<0>(p) < 0)
        cout << "......Col #" << -get<0>(p)-1 << " : " << get<1>(p) << endl;
      else
        cout << "......Row #" << get<0>(p)-1 << " : " << get<1>(p) << endl;
      q0.pop();
    }
  }


  // Repeatedly add more cover lines (greedy-algo-based)
  // Until there is no uncovered zeroes left
  int nAttempt = 1;
  while (!uncoveredZeroes.empty())
  {
    if (debug) printf("...[Attempt #%d] %zu uncovered remaining\n", nAttempt, uncoveredZeroes.size());

    // Find the next profile line with most zeroes aligned within
    if (q.size()>0)
    {
      // Take the next profile line
      Profile p = q.top();

      // Remove zeroes which are covered by this new line
      int nZeroesNewlyCovered = 0;
      vector<Point> newUncoveredZeroes;
      if (get<0>(p) < 0)
      {
        // Col
        int i = -get<0>(p)-1;
        for (auto z : uncoveredZeroes)
        {
          if (z.x != i) newUncoveredZeroes.push_back(z);
          else nZeroesNewlyCovered++;
        }

        if (nZeroesNewlyCovered>0)
        {
          coverColLines.insert(i);
          if (debug)
          {
            printf("......Added col #%d : %d more zeroes are now covered.\n", i, nZeroesNewlyCovered);
          }
        }
      }
      else
      {
        // Row
        int j = get<0>(p)-1;
        for (auto z : uncoveredZeroes)
        {
          if (z.y != j) newUncoveredZeroes.push_back(z);
          else nZeroesNewlyCovered++;
        }

        if (nZeroesNewlyCovered>0)
        {
          coverRowLines.insert(j);
          if (debug)
          {
            printf("......Added row #%d : %d more zeroes are now covered.\n", j, nZeroesNewlyCovered);
          }
        }
      }

      swap(uncoveredZeroes, newUncoveredZeroes);
      q.pop();
    }
    else
    {
      if (debug)
        cout << "...No uncovered zeroes left to collect" << endl;
      break;
    }

    nAttempt++;
  }

  if (debug)
  {
    cout << "[covered rows:]" << endl << "...";
    for (auto j : coverRowLines)
      cout << j << ",";
    cout << endl;
    cout << "[covered cols:]" << endl << "...";
    for (auto i : coverColLines)
      cout << i << ",";
    cout << endl;
  }


  // All cover lines pruned!
  return make_tuple(coverRowLines, coverColLines);
};

