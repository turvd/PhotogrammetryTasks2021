#pragma once

namespace phg {

    struct Calibration {

        cv::Vec2d project(const cv::Vec3d &point) const;
        cv::Vec3d unproject(const cv::Vec2d &pixel) const;

        cv::Matx33d K() const;

    private:
        double f;
        double cx;
        double cy;

    };

}
