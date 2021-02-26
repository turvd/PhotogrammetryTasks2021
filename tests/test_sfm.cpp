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
#include <phg/sfm/triangulation.h>

#include "utils/test_utils.h"

namespace {

    void filterMatchesF(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> keypoints_query,
                        const std::vector<cv::KeyPoint> keypoints_train, const cv::Matx33d &F, std::vector<cv::DMatch> &result, double threshold_px)
    {
        result.clear();

        for (const cv::DMatch &match : matches) {
            cv::Vec2f pt1 = keypoints_query[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints_train[match.trainIdx].pt;

            if (phg::epipolarTest(pt1, pt2, F, threshold_px)) {
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

    double matRMS(const cv::Matx33d &a, const cv::Matx33d &b)
    {
        matrix3d d = (a - b);
        d = d.mul(d);
        double rms = std::sqrt(cv::sum(d)[0] / (a.cols * a.rows));
        return rms;
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

TEST (SFM, EmatrixDecomposeSimple) {

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

    matrix34d P0, P1;
    phg::decomposeEMatrix(P0, P1, E, pts0, pts1, calib, calib);

    matrix3d R;
    R = P1.get_minor<3, 3>(0, 0);
    vector3d T;
    T(0) = P1(0, 3);
    T(1) = P1(1, 3);
    T(2) = P1(2, 3);

    matrix3d E1 = phg::composeEMatrixRT(R, T);
    matrix3d E2 = phg::composeFMatrix(P0, P1);

    EXPECT_NE(E(2, 2), 0);
    EXPECT_NE(E1(2, 2), 0);
    EXPECT_NE(E2(2, 2), 0);

    E /= E(2, 2);
    E1 /= E1(2, 2);
    E2 /= E2(2, 2);

    double rms1 = matRMS(E, E1);
    double rms2 = matRMS(E, E2);
    double rms3 = matRMS(E1, E2);

    std::cout << "E: \n" << E << std::endl;
    std::cout << "E1: \n" << E1 << std::endl;
    std::cout << "E2: \n" << E2 << std::endl;
    std::cout << "RMS1: " << rms1 << std::endl;
    std::cout << "RMS2: " << rms2 << std::endl;
    std::cout << "RMS3: " << rms3 << std::endl;

    double eps = 1e-10;
    EXPECT_LT(rms1, eps);
    EXPECT_LT(rms2, eps);
    EXPECT_LT(rms3, eps);
}

TEST (SFM, TriangulationSimple) {

    vector4d X = {0, 0, 2, 1};

    matrix34d P0 = matrix34d::eye();
    vector3d x0 = {0, 0, 1};

    // P1
    vector3d O = {2, 0, 0};
    double alpha = M_PI_4;
    double s = std::sin(alpha);
    double c = std::cos(alpha);
    matrix3d R = { c, 0, s,
                   0, 1, 0,
                  -s, 0, c};
    vector3d T = -R * O;
    matrix34d P1 = {
             R(0, 0), R(0, 1), R(0, 2), T[0],
             R(1, 0), R(1, 1), R(1, 2), T[1],
             R(2, 0), R(2, 1), R(2, 2), T[2]
    };

    // x1
    vector3d x1 = {0, 0, 1};

    std::cout << "P1:\n" << P1 << std::endl;
    std::cout << "x2:\n" << P0 * X << std::endl;
    std::cout << "x3:\n" << P1 * X << std::endl;

    matrix34d Ps[2] = {P0, P1};
    vector3d xs[2] = {x0, x1};

    vector4d X1 = phg::triangulatePoint(Ps, xs, 2);
    std::cout << "X1:\n" << X1 << std::endl;

    EXPECT_NE(X1[3], 0);
    X1 /= X1[3];

    vector4d d = X - X1;
    std::cout << "|X - X1| = " << cv::norm(d) << std::endl;

    double eps = 1e-10;
    EXPECT_LT(cv::norm(d), eps);
}

TEST (SFM, FmatrixMatchFiltering) {

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

    std::cout << "filtering matches GMS..." << std::endl;
    std::vector<DMatch> good_matches_gms;
    phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), good_matches_gms);

    std::cout << "filtering matches F..." << std::endl;
    std::vector<DMatch> good_matches_gms_plus_f;
    std::vector<DMatch> good_matches_f;
    double threshold_px = 3;
    {
        std::vector<cv::Vec2d> points1, points2;
        for (const cv::DMatch &match : good_matches) {
            cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
            points1.push_back(pt1);
            points2.push_back(pt2);
        }
        matrix3d F = phg::findFMatrix(points1, points2, threshold_px);
        filterMatchesF(good_matches, keypoints1, keypoints2, F, good_matches_f, threshold_px);
    }
    {
        std::vector<cv::Vec2d> points1, points2;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints2[match.trainIdx].pt;
            points1.push_back(pt1);
            points2.push_back(pt2);
        }
        matrix3d F = phg::findFMatrix(points1, points2, threshold_px);
        filterMatchesF(good_matches_gms, keypoints1, keypoints2, F, good_matches_gms_plus_f, threshold_px);
    }

    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_gms, "data/debug/test_sfm/matches_GMS.jpg");
    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_f, "data/debug/test_sfm/matches_F.jpg");
    drawMatches(img1, img2, keypoints1, keypoints2, good_matches_gms_plus_f, "data/debug/test_sfm/matches_GMS_plus_F.jpg");

    std::cout << "n matches gms: " << good_matches_gms.size() << std::endl;
    std::cout << "n matches F: " << good_matches_f.size() << std::endl;
    std::cout << "n matches gms + F: " << good_matches_gms_plus_f.size() << std::endl;

    EXPECT_GT(good_matches_gms_plus_f.size(), 0.5 * good_matches_gms.size());
    EXPECT_GT(good_matches_f.size(), 0.5 * good_matches_gms.size());

    EXPECT_GT(good_matches_f.size(), 0.5 * good_matches_gms_plus_f.size());
    EXPECT_GT(good_matches_gms_plus_f.size(), 0.5 * good_matches_f.size());
}
