#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <iostream>
#include <eigen3/Eigen/SVD>
#include <opencv2/calib3d.hpp>

namespace {

    void infoF(const cv::Matx33d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svdf.matrixU();
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd V = svdf.matrixV();

        std::cout << "F info:\nF:\n" << F << "\nU:\n" << U << "\ns:\n" << s << "\nV:\n" << V << std::endl;
    }

    cv::Matx33d estimateFMatrixDLT(const cv::Vec2d *m0, const cv::Vec2d *m1, int count)
    {
        int a_rows = count;
        int a_cols = 9;

        Eigen::MatrixXd A(a_rows, a_cols);

        for (int i_pair = 0; i_pair < count; ++i_pair) {

            double x0 = m0[i_pair][0];
            double y0 = m0[i_pair][1];

            double x1 = m1[i_pair][0];
            double y1 = m1[i_pair][1];

//            std::cout << "(" << x0 << ", " << y0 << "), (" << x1 << ", " << y1 << ")" << std::endl;

            A.row(i_pair) << x1*x0, x1*y0, x1, y1*x0, y1*y0, y1, x0, y0, 1.0;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd null_space = svda.matrixV().col(8);

        Eigen::MatrixXd F(3, 3);
        F.row(0) << null_space[0], null_space[1], null_space[2];
        F.row(1) << null_space[3], null_space[4], null_space[5];
        F.row(2) << null_space[6], null_space[7], null_space[8];

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svdf.matrixU();
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd V = svdf.matrixV();

        Eigen::MatrixXd S = Eigen::MatrixXd(3, 3);
        S.setZero();
        S(0, 0) = s[0];
        S(1, 1) = s[1];
        S(2, 2) = 0.0;

        F = U * S * V.transpose();

        cv::Matx33d Fcv;
        copy(F, Fcv);

        return Fcv;
    }

    // get transform matrix so that after transform centroid of points will be zero and Root Mean Square distance from centroid will be sqrt(2)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m)
    {
        cv::Vec2d centroid(0.0, 0.0);

        if (m.empty()) {
            throw std::runtime_error("Can't normalize transform");
        }

        for (int i = 0; i < (int) m.size(); ++i) {
            centroid += m[i];
        }

        centroid /= (double) m.size();

        double rms = 0;
        for (int i = 0; i < (int) m.size(); ++i) {
            cv::Vec2d d = m[i] - centroid;
            rms += d[0] * d[0] + d[1] * d[1];
        }
        rms = std::sqrt(rms / (2 * m.size()));

        if (rms == 0) {
            throw std::runtime_error("Can't normalize transform");
        }

        double s = std::sqrt(2) / rms;

        std::cout << "NORMALIZE TRANSFORM: centroid = " << centroid << ", scale = " << s << std::endl;

        cv::Matx33d T(s, 0.0, -s*centroid[0],
                      0.0, s, -s*centroid[1],
                      0.0, 0.0, 1.0);

        return T;
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixRANSAC: m0.size() != m1.size()");
        }

        const int n_matches = m0.size();

        cv::Matx33d TN0 = getNormalizeTransform(m0);
        cv::Matx33d TN1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0_t(n_matches);
        std::vector<cv::Vec2d> m1_t(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            m0_t[i] = transformPoint(m0[i], TN0);
            m1_t[i] = transformPoint(m1[i], TN1);
        }

        {
//             check log: centroid should become close to zero, scale close to 1
            getNormalizeTransform(m0_t);
            getNormalizeTransform(m1_t);
        }

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = 10000;

        const int n_samples = 8;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx33d best_F;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_matches, n_samples, &seed);

            cv::Vec2d ms0[n_samples];
            cv::Vec2d ms1[n_samples];
            std::vector<cv::Vec2d> AAA;
            std::vector<cv::Vec2d> BBB;
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = m0_t[sample[i]];
                ms1[i] = m1_t[sample[i]];
                AAA.push_back(ms0[i]);
                BBB.push_back(ms1[i]);
            }

            cv::Matx33d F = estimateFMatrixDLT(ms0, ms1, n_samples);

            // denormalize
            F = TN1.t() * F * TN0;

            int support = 0;
            for (int i = 0; i < n_matches; ++i) {
                //                    todo todo todo                                                         todo todo todo
                if (phg::epipolarTest(m0[i], m1[i], F, threshold_px) && phg::epipolarTest(m1[i], m0[i], F.t(), threshold_px))
                {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_F = F;

                std::cout << "estimateFMatrixRANSAC : support: " << best_support << "/" << n_matches << std::endl;
                infoF(F);

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        std::cout << "estimateFMatrixRANSAC : best support: " << best_support << "/" << n_matches << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateFMatrixRANSAC : failed to estimate fundamental matrix");
        }

        return best_F;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector <cv::Vec2d> &m0, const std::vector <cv::Vec2d> &m1, double threshold_px) {
    return estimateFMatrixRANSAC(m0, m1, threshold_px);
}

cv::Matx33d phg::findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px) {
    return cv::findFundamentalMat(m0, m1, cv::FM_RANSAC, threshold_px);
}
