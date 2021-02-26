#pragma once

#include <opencv2/core.hpp>
#include <phg/core/calibration.h>

namespace phg {

    cv::Matx33d fmatrix2ematrix(const cv::Matx33d &F, const Calibration &calib0, const Calibration &calib1);

    void decomposeEMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &E);

}