#include "ematrix.h"

#include "defines.h"
#include "fmatrix.h"

#include <eigen3/Eigen/SVD>

namespace {

    // essential matrix must have exactly two equal non zero singular values
    void ensureSpectralProperty(matrix3d &Ecv)
    {
        Eigen::MatrixXd E;
        copy(Ecv, E);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svd.matrixU();
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd V = svd.matrixV();

        Eigen::MatrixXd S = Eigen::MatrixXd(3, 3);
        S.setZero();

        S(0, 0) = 1.0;
        S(1, 1) = 1.0;
        S(2, 2) = 0.0;

        E = U * S * V.transpose();

        copy(E, Ecv);
    }

}

cv::Matx33d phg::fmatrix2ematrix(const cv::Matx33d &F, const phg::Calibration &calib0, const phg::Calibration &calib1)
{
    // TODO check that in metashape converting f matrix to ematrix ith non trivial callib works
    // TODO check that in metashape converting f matrix to ematrix ith non trivial callib works
    // TODO check that in metashape converting f matrix to ematrix ith non trivial callib works
    // TODO check that in metashape converting f matrix to ematrix ith non trivial callib works
    // TODO check that in metashape converting f matrix to ematrix ith non trivial callib works


    // TODO check that correct conversion (transpose ok)

    // TODO add separate test for fmatrix (check 1) reprojection errors 2) f matrix singular value property)
    // TODO add separate test for ematrix (check 1) reprojection errors of normalized coordinates 2) e matrix singular value property)
    // TODO add separate test for ematrix decomposition: 1) check angle between cameras 2) check translation direction between cameras

    matrix3d E = calib1.K().t() * F * calib0.K();

    ensureSpectralProperty(E);

    return E;
}

void phg::decomposeEMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &E)
{
    // find 4 solutions
    // implement dlt triangulation
    // test each of 4 solutions
}
