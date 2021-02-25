#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <libutils/timer.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>

#include "utils/test_utils.h"

namespace {

    void filterMatchesF(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> keypoints_query,
                        const std::vector<cv::KeyPoint> keypoints_train, const cv::Matx33d &F, std::vector<cv::DMatch> &result)
    {
        result.clear();

        double reprojection_error_threshold_px = 3;

        for (const cv::DMatch &match : matches) {
            cv::Vec2f pt1 = keypoints_query[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints_train[match.trainIdx].pt;

            if (phg::epipolarTest(pt1, pt2, F, reprojection_error_threshold_px)) {
                result.push_back(match);
            }
        }
    }

}

TEST (MATCHING, Test2View) {

    using namespace cv;

    cv::Mat img1 = cv::imread("data/src/test_sfm/saharov/IMG_3023.JPG");
    cv::Mat img2 = cv::imread("data/src/test_sfm/saharov/IMG_3024.JPG");

    std::cout << "detecting points..." << std::endl;
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

    std::cout << "matching points..." << std::endl;
    std::vector<std::vector<DMatch>> knn_matches;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::vector<DMatch> good_matches(knn_matches.size());
    for (int i = 0; i < (int) knn_matches.size(); ++i) {
        good_matches[i] = knn_matches[i][0];
    }

    std::cout << "filtering points..." << std::endl;
    {
        std::vector<DMatch> tmp;
        phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), tmp);
        std::swap(tmp, good_matches);
    }

    std::vector<cv::Vec2d> points1, points2_correct;
    for (const cv::DMatch &match : good_matches) {
        cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
        points1.push_back(pt1);
        points2_correct.push_back(pt2);
    }

    drawMatches(img1, img2, keypoints1, keypoints2, good_matches, "data/debug/test_sfm/matches.jpg");

    auto F = phg::findFMatrix(points1, points2_correct);
    {
        std::vector<DMatch> tmp;
        filterMatchesF(good_matches, keypoints1, keypoints2, F, tmp);
        std::swap(tmp, good_matches);
    }
    drawMatches(img1, img2, keypoints1, keypoints2, good_matches, "data/debug/test_sfm/matchesF.jpg");

//    std::cout << "finding F matrix..." << std::endl;
//    cv::Matx33d F = phg::findFMatrix(points1, points2);
//    std::cout << F << std::endl;
//
//    std::cout << "finding camera matrices..." << std::endl;
//    cv::Matx34d P1, P2;
//    phg::decomposeFMatrix(P1, P2, F);
//
//    std::cout << "P1: " << P1 << std::endl;
//    std::cout << "P2: " << P2 << std::endl;
}
