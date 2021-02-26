#pragma once

#include <opencv2/core.hpp>
#include <eigen3/Eigen/Core>

typedef cv::Matx22d matrix2d;
typedef cv::Vec2d vector2d;
typedef cv::Matx33d matrix3d;
typedef cv::Vec3d vector3d;
typedef cv::Matx44d matrix4d;
typedef cv::Vec4d vector4d;

inline void copy(const matrix3d &Fcv, Eigen::MatrixXd &F)
{
    F = Eigen::MatrixXd(3, 3);

    F(0, 0) = Fcv(0, 0); F(0, 1) = Fcv(0, 1); F(0, 2) = Fcv(0, 2);
    F(1, 0) = Fcv(1, 0); F(1, 1) = Fcv(1, 1); F(1, 2) = Fcv(1, 2);
    F(2, 0) = Fcv(2, 0); F(2, 1) = Fcv(2, 1); F(2, 2) = Fcv(2, 2);
}

inline void copy(const Eigen::MatrixXd &F, matrix3d &Fcv)
{
    Fcv(0, 0) = F(0, 0); Fcv(0, 1) = F(0, 1); Fcv(0, 2) = F(0, 2);
    Fcv(1, 0) = F(1, 0); Fcv(1, 1) = F(1, 1); Fcv(1, 2) = F(1, 2);
    Fcv(2, 0) = F(2, 0); Fcv(2, 1) = F(2, 1); Fcv(2, 2) = F(2, 2);
}
