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

#ifndef HDF5OPENCV_KVHWT01S
#define HDF5OPENCV_KVHWT01S

#include <stdexcept>
#include <exception>
#include <string>

#include "opencv/cv.h"

#include "hdf5/hdf5.h"

#if (H5_VERS_MINOR==6)
#include "H5LT.h"
#else
#include "hdf5/hdf5_hl.h"
#endif

namespace hdf5opencv
{

class Hdf5OpenCVException : public std::runtime_error {
public:
  Hdf5OpenCVException(const std::string& what_arg) :
      std::runtime_error(what_arg) { }
};

void hdf5create(const char *filename,
                bool overwrite = false);

void hdf5save(const char * filename,
              const char * dataset_name,
              cv::Mat& dataset,
              bool overwrite = false);

void hdf5save(const char * filename,
              const char * dataset_name,
              const char * strbuf,
              bool overwrite = false);

void hdf5load(const char * filename,
              const char * dataset_name,
              cv::Mat& dataset);

void hdf5load(const char * filename,
              const char * dataset_name,
              char ** strbuf);

} /* hdf5opencv */

#endif /* end of include guard: HDF5OPENCV_KVHWT01S */
