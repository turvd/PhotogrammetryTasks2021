#pragma once

#include <opencv2/core.hpp>


namespace phg {

    cv::Vec3d findPointNViewEstimate(const cv::Matx34d *Ps, const cv::Vec2d *ms, int count);

}
