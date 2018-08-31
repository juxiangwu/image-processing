//  Copyright (C) 2010 Daniel Maturana
//  This file is part of hdf5opencv.
//
//  hdf5opencv is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  hdf5opencv is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with hdf5opencv. If not, see <http://www.gnu.org/licenses/>.
//

#include <iostream>
#include <boost/filesystem.hpp>

#include "opencv/cv.h"
#include "hdf5opencv.hh"

template <class T>
void print_mat(cv::Mat_<T>& m) {
  for (int i=0; i<m.rows; ++i) {
    for (int j=0; j<m.cols; ++j) {
      std::cout << m(i,j) << " ";
    }
    std::cout << std::endl;
  }
}

template <>
void print_mat(cv::Mat_<unsigned char>& m) {
  for (int i=0; i<m.rows; ++i) {
    for (int j=0; j<m.cols; ++j) {
      std::cout << static_cast<int>(m(i,j)) << " ";
    }
    std::cout << std::endl;
  }
}

template <class T>
void test() {
  cv::Mat_<T> a(cv::Size(2, 8));
  for (int i = 0; i < a.rows; ++i) {
    for (int j = 0; j < a.cols; ++j) {
      a(i,j) = (a.cols*i) + j;
    }
  }
  std::cout << "\nMatrix to write:\n";
  print_mat(a);
  std::cout << "\nWriting\n";
  // overwrite over old "test.h5" file
  hdf5opencv::hdf5save("test.h5", "/data", a, true);
  std::cout << "Reading\n";
  cv::Mat_<T> b;
  hdf5opencv::hdf5load("test.h5", "/data", b);
  std::cout << "\nRead matrix:\n";
  print_mat(b);
}

int main (int argc, char const* argv[]) {

  std::cout << "\n*** FLOAT TEST ***\n";
  test<float>();

  std::cout << "\n*** DOUBLE TEST ***\n";
  test<double>();

  std::cout << "\n*** INT TEST ***\n";
  test<int>();

  std::cout << "\n*** CHAR TEST ***\n";
  test<unsigned char>();

  std::cout << "\n*** string TEST ***\n";
  const char *strbuf = "hello world blabla";
  std::cout << "Writing " << strbuf << std::endl;
  hdf5opencv::hdf5save("test.h5", "/hellostr", strbuf, true);
  char *outbuf;
  hdf5opencv::hdf5load("test.h5", "/hellostr", &outbuf);
  std::cout << "Read " << outbuf << std::endl;
  return 0;
}
