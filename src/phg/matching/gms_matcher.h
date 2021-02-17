#pragma once

#include <opencv2/core.hpp>

namespace phg {

    void filterMatchesGMS(const std::vector<cv::DMatch> &matches,
                                      const std::vector<cv::KeyPoint> keypoints_query,
                                      const std::vector<cv::KeyPoint> keypoints_train,
                                      const cv::Size &sz_query,
                                      const cv::Size &sz_train,
                                      std::vector<cv::DMatch> &filtered_matches);

}
