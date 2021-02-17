#pragma once

#include <opencv2/core.hpp>

namespace phg {

    struct DescriptorMatcher {

        virtual void train(const cv::Mat &train_desc) = 0;
        virtual void knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const = 0;

        static void filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &filtered_matches);

        static void filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                          const std::vector<cv::KeyPoint> keypoints_query,
                                          const std::vector<cv::KeyPoint> keypoints_train,
                                          std::vector<cv::DMatch> &filtered_matches);

    };


//    template <typename MATCHER> // template to enable support for OpenCV matchers
//    std::vector<cv::DMatch> matchDescriptors(const MATCHER &matcher, const cv::Mat &desc0, const cv::Mat &desc1)
//    {
//        std::vector<std::vector<cv::DMatch>> matches;
//        matcher.knnMatch(desc0, desc1, matches, 2);
//
//        std::vector<cv::DMatch> filtered_matches_ratio;
//        DescriptorMatcher::filterMatchesRatioTest(matches, filtered_matches_ratio);
//
//        std::vector<cv::DMatch> filtered_matches_clusters;
//        DescriptorMatcher::filterMatchesClusters(filtered_matches_ratio, filtered_matches_clusters);
//
//        // todo add left-right check ?
//
//        return filtered_matches_clusters;
//    }

}