#pragma once

#include <opencv2/core.hpp>

namespace phg {

    cv::Matx33d findFMatrix(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1);
    cv::Matx33d findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1);


    void decomposeFMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &F);

    bool epipolarTest(const cv::Vec2f &pt0, const cv::Vec2f &pt1, const cv::Matx33d &F, double t);

}