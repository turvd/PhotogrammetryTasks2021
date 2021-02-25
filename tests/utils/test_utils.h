#pragma once

#include <opencv2/core.hpp>

cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1);

std::string getTestName();

std::string getTestSuiteName();

void drawMatches(const cv::Mat &img1,
                 const cv::Mat &img2,
                 const std::vector<cv::KeyPoint> &keypoints1,
                 const std::vector<cv::KeyPoint> &keypoints2,
                 const std::vector<cv::DMatch> &matches,
                 const std::string &path);
