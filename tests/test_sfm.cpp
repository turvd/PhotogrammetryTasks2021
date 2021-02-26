#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <libutils/timer.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>
#include <phg/sfm/ematrix.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/sfm/defines.h>
#include <eigen3/Eigen/SVD>

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

    // Fundamental matrix has to be of rank 2. See Hartley & Zisserman, p.243
    bool checkFmatrixSpectralProperty(const matrix3d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();

        std::cout << "checkFmatrixSpectralProperty: s: " << s.transpose() << std::endl;

        double thresh = 1e10;
        return s[0] > thresh * s[2] && s[1] > thresh * s[2];
    }

    // Essential matrix has to be of rank 2, and two non-zero singular values have to be equal. See Hartley & Zisserman, p.257
    bool checkEmatrixSpectralProperty(const matrix3d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();

        std::cout << "checkEmatrixSpectralProperty: s: " << s.transpose() << std::endl;

        double thresh = 1e10;

        bool rank2 = s[0] > thresh * s[2] && s[1] > thresh * s[2];
        bool equal = (s[0] < (1.0 + thresh) * s[1]) && (s[1] < (1.0 + thresh) * s[0]);

        return rank2 && equal;
    }

}

#define TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps) \
EXPECT_FALSE(phg::epipolarTest(pt0, pt1, F, std::max(0.0, t - eps))); \
EXPECT_TRUE(phg::epipolarTest(pt0, pt1, F, t + eps));

TEST (SFM, EpipolarDist) {

    const vector2d pt0 = {0, 0};
    const double eps = 1e-5;

    {
        // line: y = 0
        const double l[3] = {0, 1, 0};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {1000, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {0, 1000};
        t = 1000;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);
    }

    {
        // line: y = x
        const double l[3] = {1, -1, 0};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {1, 1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {-1, -1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {-1, 1};
        t = std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {10, 0};
        t = 10 / std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);
    }

    {
        // line: y = x + 1
        const double l[3] = {1, -1, 1};
        const matrix3d F = {0, 0, l[0], 0, 0, l[1], 0, 0, l[2]};

        vector2d pt1;
        double t;

        pt1 = {0, 1};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {1, 2};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {-1, 0};
        t = 0;
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {-1, 2};
        t = std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);

        pt1 = {10, 1};
        t = 10 / std::sqrt(2);
        TEST_EPIPOLAR_LINE(pt0, pt1, F, t, eps);
    }
}

TEST (SFM, FmatrixSimple) {

    std::vector<cv::Vec2d> pts0, pts1;
    std::srand(1);
    for (int i = 0; i < 8; ++i) {
        pts0.push_back({(double) (std::rand() % 100), (double) (std::rand() % 100)});
        pts1.push_back({(double) (std::rand() % 100), (double) (std::rand() % 100)});
    }

    matrix3d F = phg::findFMatrix(pts0, pts1);
    matrix3d Fcv = phg::findFMatrixCV(pts0, pts1);

    EXPECT_TRUE(checkFmatrixSpectralProperty(F));
    EXPECT_TRUE(checkFmatrixSpectralProperty(Fcv));
}

TEST (SFM, EmatrixSimple) {

    phg::Calibration calib(360, 240);
    std::cout << "EmatrixSimple: calib: \n" << calib.K() << std::endl;

    std::vector<cv::Vec2d> pts0, pts1;
    std::srand(1);
    for (int i = 0; i < 8; ++i) {
        pts0.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
        pts1.push_back({(double) (std::rand() % calib.width()), (double) (std::rand() % calib.height())});
    }

    matrix3d F = phg::findFMatrix(pts0, pts1, 10);
    matrix3d E = phg::fmatrix2ematrix(F, calib, calib);

    EXPECT_TRUE(checkEmatrixSpectralProperty(E));
}

TEST (MATCHING, Test2View) {

    return;

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
