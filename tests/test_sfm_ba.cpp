#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <fstream>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <phg/matching/gms_matcher.h>
#include <phg/sfm/fmatrix.h>
#include <phg/sfm/ematrix.h>
#include <phg/sfm/sfm_utils.h>
#include <phg/sfm/defines.h>
#include <phg/sfm/triangulation.h>
#include <phg/sfm/resection.h>
#include <phg/utils/point_cloud_export.h>

#include <ceres/rotation.h>
#include <ceres/ceres.h>

#include "utils/test_utils.h"

#define FIX_INTRINSICS_CALIBRATION 1


namespace {

    vector3d relativeOrientationAngles(const matrix3d &R0, const vector3d &O0, const matrix3d &R1, const vector3d &O1) {
        vector3d a = R0 * vector3d{0, 0, 1};
        vector3d b = O0 - O1;
        vector3d c = R1 * vector3d{0, 0, 1};

        double norma = cv::norm(a);
        double normb = cv::norm(b);
        double normc = cv::norm(c);

        if (norma == 0 || normb == 0 || normc == 0) {
            throw std::runtime_error("norma == 0 || normb == 0 || normc == 0");
        }

        a /= norma;
        b /= normb;
        c /= normc;

        vector3d cos_vals;

        cos_vals[0] = a.dot(c);
        cos_vals[1] = a.dot(b);
        cos_vals[2] = b.dot(c);

        return cos_vals;
    }

    void transform(matrix3d &R, vector3d &O) {
        matrix4d H = matrix4d::diag({1, -1, -1, 1});
        matrix3d Rinv = H.inv().get_minor<3, 3>(0, 0);

        auto tmp = H * vector4d({O[0], O[1], O[2], 1.0});
        O = {tmp[0] / tmp[3], tmp[1] / tmp[3], tmp[2] / tmp[3]};
        R = R * Rinv;
    }

    // one track corresponds to one 3d point
    struct Track {
        std::vector<std::pair<int, int>> img_kpt_pairs;
    };

}

void generateTiePointsCloud(const std::vector<vector3d> tie_points,
                            const std::vector<Track> tracks,
                            const std::vector<std::vector<cv::KeyPoint>> keypoints,
                            const std::vector<cv::Mat> imgs,
                            const std::vector<char> aligned,
                            const std::vector<matrix34d> cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors);

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose=false);

TEST (SFM, ReconstructNViews) {
    using namespace cv;

    // Мы используем камеры из датасета temple47
    // Чтобы было проще - упорядочим их заранее в файле data/src/datasets/temple47/ordered_filenames.txt
    // Камеры templeR0001 и templeR0030 - почти совпадают, поэтому не будем работать с templeR0001 (осталось 46 камер)
    std::vector<cv::Mat> imgs;
    std::vector<std::string> imgs_labels;
    {
        std::ifstream in("data/src/datasets/temple47/ordered_filenames.txt");
        size_t nimages = 0;
        in >> nimages;
        std::cout << nimages << " images" << std::endl;
        for (int i = 0; i < nimages; ++i) {
            std::string img_name;
            in >> img_name;
            std::string img_path = std::string("data/src/datasets/temple47/") + img_name;
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) {
                throw std::runtime_error("Can't read image: " + to_string(img_path));
            }
            imgs.push_back(img);
            imgs_labels.push_back(img_name);
        }
        // Если хочется попробовать другие камеры - можно начать с конца:
        std::reverse(imgs.begin(), imgs.end());
        std::reverse(imgs_labels.begin(), imgs_labels.end());
    }

    phg::Calibration calib(imgs[0].cols, imgs[0].rows);
    if (FIX_INTRINSICS_CALIBRATION) {
        calib.f_ = 1520.4; // see temple47/README.txt about K-matrix (i.e. focal length = K11 from templeR_par.txt)
    }
    for (const auto &img : imgs) {
        rassert(img.cols == imgs[0].cols && img.rows == imgs[0].rows, 34125412512512);
    }

    const int n_imgs = 5;//imgs.size();

    std::cout << "detecting points..." << std::endl;
    std::vector<std::vector<cv::KeyPoint>> keypoints(n_imgs);
    std::vector<std::vector<int>> track_ids(n_imgs);
    std::vector<cv::Mat> descriptors(n_imgs);
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    for (int i = 0; i < (int) n_imgs; ++i) {
        detector->detectAndCompute(imgs[i], cv::noArray(), keypoints[i], descriptors[i]);
        track_ids[i].resize(keypoints[i].size(), -1);
    }

    std::cout << "matching points..." << std::endl;
    using Matches = std::vector<cv::DMatch>;
    std::vector<std::vector<Matches>> matches(n_imgs);
    #pragma omp parallel for
    for (int i = 0; i < n_imgs; ++i) {
        matches[i].resize(n_imgs);
        for (int j = 0; j < n_imgs; ++j) {
            if (i == j) {
                continue;
            }

            // Flann matching
            std::vector<std::vector<DMatch>> knn_matches;
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->knnMatch( descriptors[i], descriptors[j], knn_matches, 2 );
            std::vector<DMatch> good_matches(knn_matches.size());
            for (int k = 0; k < (int) knn_matches.size(); ++k) {
                good_matches[k] = knn_matches[k][0];
            }

            // Filtering matches GMS
            std::vector<DMatch> good_matches_gms;
            int inliers = phg::filterMatchesGMS(good_matches, keypoints[i], keypoints[j], imgs[i].size(), imgs[j].size(), good_matches_gms, false);
            #pragma omp critical
            {
                std::cout << "Cameras " << i << "-" << j << " (" << imgs_labels[i] << "-" << imgs_labels[j] << "): " << inliers << " matches" << std::endl;
            }

            matches[i][j] = good_matches_gms;
        }
    }

    std::vector<Track> tracks;
    std::vector<vector3d> tie_points;
    std::vector<matrix34d> cameras(n_imgs);
    std::vector<char> aligned(n_imgs);

    // align first two cameras
    {
        std::cout << "Initial alignment from cameras #0 and #1 (" << imgs_labels[0] << ", " << imgs_labels[1] << ")" << std::endl;
        // matches from first to second image in specified sequence
        const Matches &good_matches_gms = matches[0][1];
        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[0];
        const std::vector<cv::KeyPoint> &keypoints1 = keypoints[1];
        const phg::Calibration &calib0 = calib;
        const phg::Calibration &calib1 = calib;

        std::vector<cv::Vec2d> points0, points1;
        for (const cv::DMatch &match : good_matches_gms) {
            cv::Vec2f pt1 = keypoints0[match.queryIdx].pt;
            cv::Vec2f pt2 = keypoints1[match.trainIdx].pt;
            points0.push_back(pt1);
            points1.push_back(pt2);
        }

        matrix3d F = phg::findFMatrix(points0, points1, 3, false);
        matrix3d E = phg::fmatrix2ematrix(F, calib0, calib1);

        matrix34d P0, P1;
        phg::decomposeEMatrix(P0, P1, E, points0, points1, calib0, calib1, false);

        {
            matrix3d R0, R1;
            vector3d O0, O1;
            phg::decomposeUndistortedPMatrix(R0, O0, P0);
            phg::decomposeUndistortedPMatrix(R1, O1, P1);
            transform(R0, O0);
            transform(R1, O1);
            P0 = phg::composeCameraMatrixRO(R0, O0);
            P1 = phg::composeCameraMatrixRO(R1, O1);
        }

        cameras[0] = P0;
        cameras[1] = P1;
        aligned[0] = true;
        aligned[1] = true;

        matrix34d Ps[2] = {P0, P1};
        for (int i = 0; i < (int) good_matches_gms.size(); ++i) {
            vector3d ms[2] = {calib0.unproject(points0[i]), calib1.unproject(points1[i])};
            vector4d X = phg::triangulatePoint(Ps, ms, 2);

            if (X(3) == 0) {
                std::cerr << "infinite point" << std::endl;
                continue;
            }

            vector3d X3d{X(0) / X(3), X(1) / X(3), X(2) / X(3)};

            tie_points.push_back(X3d);

            Track track;
            track.img_kpt_pairs.push_back({0, good_matches_gms[i].queryIdx});
            track.img_kpt_pairs.push_back({1, good_matches_gms[i].trainIdx});
            track_ids[0][good_matches_gms[i].queryIdx] = tracks.size();
            track_ids[1][good_matches_gms[i].trainIdx] = tracks.size();
            tracks.push_back(track);
        }

        int ncameras = 2;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, "data/debug/test_sfm_ba/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, "data/debug/test_sfm_ba/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }

    // append remaining cameras one by one
    for (int i_camera = 2; i_camera < n_imgs; ++i_camera) {

        const std::vector<cv::KeyPoint> &keypoints0 = keypoints[i_camera];
        const phg::Calibration &calib0 = calib;

        std::vector<vector3d> Xs;
        std::vector<vector2d> xs;
        int neighbours_limit = 2;
        for (int i_camera_prev = i_camera - neighbours_limit; i_camera_prev < i_camera; ++i_camera_prev) {
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id != -1) {
                    Xs.push_back(tie_points[track_id]);
                    cv::Vec2f pt = keypoints0[match.queryIdx].pt;
                    xs.push_back(pt);
                }
            }
        }

        std::cout << "Append camera #" << i_camera << " (" << imgs_labels[i_camera] << ") to alignment via " << Xs.size() << " common points" << std::endl;
        matrix34d P = phg::findCameraMatrix(calib0, Xs, xs, false);

        cameras[i_camera] = P;
        aligned[i_camera] = true;

        for (int i_camera_prev = i_camera - neighbours_limit; i_camera_prev < i_camera; ++i_camera_prev) {
            const std::vector<cv::KeyPoint> &keypoints1 = keypoints[i_camera_prev];
            const phg::Calibration &calib1 = calib;
            const Matches &good_matches_gms = matches[i_camera][i_camera_prev];
            for (const cv::DMatch &match : good_matches_gms) {
                int track_id = track_ids[i_camera_prev][match.trainIdx];
                if (track_id == -1) {
                    matrix34d Ps[2] = {P, cameras[i_camera_prev]};
                    cv::Vec2f pts[2] = {keypoints0[match.queryIdx].pt, keypoints1[match.trainIdx].pt};
                    vector3d ms[2] = {calib0.unproject(pts[0]), calib1.unproject(pts[1])};
                    vector4d X = phg::triangulatePoint(Ps, ms, 2);

                    if (X(3) == 0) {
                        std::cerr << "infinite point" << std::endl;
                        continue;
                    }

                    tie_points.push_back({X(0) / X(3), X(1) / X(3), X(2) / X(3)});

                    Track track;
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track.img_kpt_pairs.push_back({i_camera_prev, match.trainIdx});
                    track_ids[i_camera][match.queryIdx] = tracks.size();
                    track_ids[i_camera_prev][match.trainIdx] = tracks.size();
                    tracks.push_back(track);
                } else {
                    Track &track = tracks[track_id];
                    track.img_kpt_pairs.push_back({i_camera, match.queryIdx});
                    track_ids[i_camera][match.queryIdx] = track_id;
                }
            }
        }

        int ncameras = i_camera + 1;

        std::vector<vector3d> tie_points_and_cameras;
        std::vector<cv::Vec3b> tie_points_colors;
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, "data/debug/test_sfm_ba/point_cloud_" + to_string(ncameras) + "_cameras.ply", tie_points_colors);

        runBA(tie_points, tracks, keypoints, cameras, ncameras, calib);
        generateTiePointsCloud(tie_points, tracks, keypoints, imgs, aligned, cameras, ncameras, tie_points_and_cameras, tie_points_colors);
        phg::exportPointCloud(tie_points_and_cameras, "data/debug/test_sfm_ba/point_cloud_" + to_string(ncameras) + "_cameras_ba.ply", tie_points_colors);
    }
}

class ReprojectionError {
public:
    ReprojectionError(double x, double y) : observed_x(x), observed_y(y)
    {}

    template <typename T>
    bool operator()(const T* camera_extrinsics, // положение камеры:   [6] = {rotation[3], translation[3]} (разное для всех кадров, т.к. каждая фотография со своего ракурса)
                    const T* camera_intrinsics, // внутренние калибровочные параметры камеры: [5] = {k1, k2, f, cx, cy} (одни и те же для всех кадров, т.к. снято на одну и ту же камеру)
                    const T* point,             // 3D точка: [3]  = {x, y, z}
                    T* residuals) const {       // невязка:  [2]  = {dx, dy}
        // rotation[3] - angle-axis rotation, поворачиваем точку point->p (чтобы перейти в локальную систему координат камеры)
        // подробнее см. https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
        // (P.S. у камеры всмысле вращения три степени свободы)
        T p[3];
        const T *camera_angle_axis_rotation = camera_extrinsics; camera_extrinsics += 3;
        ceres::AngleAxisRotatePoint(camera_angle_axis_rotation, point, p);

        // translation[3] - сдвиг в локальную систему координат камеры
        const T *camera_translation = camera_extrinsics; camera_extrinsics += 3;
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        // Проецируем точку на плоскость матрицы (т.е. плоскость Z=1)
        T x = p[0] / p[2];
        T y = p[1] / p[2];

        // k1, k2 - коэффициенты радиального искажения (radial distortion)
        const T *camera_k1_k2 = camera_intrinsics; camera_intrinsics += 2;
//        T k1 = camera_k1_k2[0];
//        T k2 = camera_k1_k2[1];
//        T r2 = x * x + y * y;
//        T r4 = x * x + y * y;
//        x = x * (1.0 + k1 * r2 + k2 * r4);
//        y = y * (1.0 + k1 * r2 + k2 * r4);

        // Переводим из координат плоскости матрицы (Z=1) в пиксели (т.е. на плоскость Z=фокальная длина)
        // camera[8] = f
        const T focal = camera_intrinsics[0]; camera_intrinsics += 1;
        x = focal * x;
        y = focal * y;

        // Из координат когда точка (0, 0) - центр оптической оси
        // Переходим в координаты когда точка (0, 0) - левый верхний угол картинки
        // camera[9, 10] = cx, cy - координаты центра оптической оси (обычно это центр картинки, но часто он чуть смещен)
        const T *camera_cx_cy = camera_intrinsics; camera_intrinsics += 2;
        x += camera_cx_cy[0];
        y += camera_cx_cy[1];

        // Нашли невязку репроекции
        residuals[0] = x - observed_x;
        residuals[1] = y - observed_y;

        return true;
    }
protected:
    double observed_x;
    double observed_y;
};

void printCamera(double* camera_intrinsics)
{
    std::cout << "camera: k1=" << camera_intrinsics[0] << ", k2=" << camera_intrinsics[1] << ", "
              << "f=" << camera_intrinsics[2] << ", "
              << "cx=" << camera_intrinsics[3] << ", cy=" << camera_intrinsics[4] << std::endl;
}

void runBA(std::vector<vector3d> &tie_points,
           std::vector<Track> &tracks,
           std::vector<std::vector<cv::KeyPoint>> &keypoints,
           std::vector<matrix34d> &cameras,
           int ncameras,
           phg::Calibration &calib,
           bool verbose)
{
    // Формулируем задачу
    ceres::Problem problem;

    // внутренние калибровочные параметры камеры: [5] = {k1, k2, f, cx, cy}
    double camera_intrinsics[5] = {0.0, 0.0, calib.f_, calib.cx_ + 0.5 * calib.width(), calib.cy_ + 0.5 * calib.height()};
    std::cout << "Before BA ";
    printCamera(camera_intrinsics);

    const int CAMERA_EXTRINSICS_NPARAMS = 6;

    // внешние калибровочные параметры камеры для каждого кадра: [6] = {rotation[3], translation[3]}
    std::vector<double> cameras_extrinsics(CAMERA_EXTRINSICS_NPARAMS * ncameras, 0.0);
    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* rotation_angle_axis = camera_extrinsics + 0;
        double* translation = camera_extrinsics + 3;

        ceres::RotationMatrixToAngleAxis(&(R(0, 0)), rotation_angle_axis);
        for (int d = 0; d < 3; ++d) {
            translation[d] = O[d];
        }
    }

    // остались только блоки параметров для 3D точек, но их аллоцировать не обязательно, т.к. мы можем их оптимизировать напрямую в tie_points массиве

    const double sigma = 2.0; // измеряется в пикселях

    double inliers_mse = 0.0;
    size_t inliers = 0;
    size_t nprojections = 0;

    std::vector<ceres::CostFunction*> reprojection_residuals;

    // Создаем невязки для всех проекций 3D точек в камеры (т.е. для всех наблюдений этих ключевых точек)
    for (size_t i = 0; i < tie_points.size(); ++i) {
        const Track &track = tracks[i];
        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;
            int keypoint_id = track.img_kpt_pairs[ci].second;
            cv::Vec2f px = keypoints[camera_id][keypoint_id].pt;

            ceres::CostFunction* keypoint_reprojection_residual = new ceres::AutoDiffCostFunction<ReprojectionError,
                    2, // количество невязок (размер искомого residual массива переданного в функтор, т.е. размерность искомой невязки, у нас это dx, dy (ошибка проекции по обеим осям)
                    6, 5, 3> // число параметров в каждом блоке параметров, у нас три блок параметров (внешние параметры камеры[6], внутренние параметры камеры[5] и 3D точка)
                    (new ReprojectionError(px[0], px[1]));
            reprojection_residuals.push_back(keypoint_reprojection_residual);

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;

            // блоки параметров для 3D точек аллоцировать не обязательно, т.к. мы можем их оптимизировать напрямую в tie_points массиве
            double* point3d_params = &(tie_points[i][0]);

            {
                const double* params[3];
                double residual[2] = {-1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < 3.0 * sigma) {
                    inliers_mse += error2;
                    ++inliers;
                }
                ++nprojections;
            }

            problem.AddResidualBlock(keypoint_reprojection_residual, new ceres::HuberLoss(3.0 * sigma),
                                     camera_extrinsics,
                                     camera_intrinsics,
                                     point3d_params);
        }
    }
    std::cout << "After BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;

    if (FIX_INTRINSICS_CALIBRATION) {
        // Полностью фиксируем внутренние калибровочные параметры камеры (общие для всех кадров)
        problem.SetParameterBlockConstant(camera_intrinsics);
    }
    {
        // Полностью фиксируем положение первой камеры (чтобы не уползло облако точек)
        size_t camera_id = 0;
        double* camera0_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetParameterBlockConstant(camera0_extrinsics);
    }
    {
        // Фиксируем координаты второй камеры, т.е. translation[3] (чтобы фиксировать масштаб)
        size_t camera_id = 1;
        double* camera1_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        problem.SetParameterization(camera1_extrinsics, new ceres::SubsetParameterization(6, {3, 4, 5}));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = verbose;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (verbose) {
        std::cout << summary.BriefReport() << std::endl;
    }

    inliers_mse = 0.0;
    inliers = 0;
    nprojections = 0;
    size_t next_loss_k = 0;
    // Создаем невязки для всех проекций 3D точек в камеры (т.е. для всех наблюдений этих ключевых точек)
    for (size_t i = 0; i < tie_points.size(); ++i) {
        const Track &track = tracks[i];
        for (size_t ci = 0; ci < track.img_kpt_pairs.size(); ++ci) {
            int camera_id = track.img_kpt_pairs[ci].first;

            ceres::CostFunction* keypoint_reprojection_residual = reprojection_residuals[next_loss_k++];

            double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;

            // блоки параметров для 3D точек аллоцировать не обязательно, т.к. мы можем их оптимизировать напрямую в tie_points массиве
            double* point3d_params = &(tie_points[i][0]);

            {
                const double* params[3];
                double residual[2] = {-1.0};
                params[0] = camera_extrinsics;
                params[1] = camera_intrinsics;
                params[2] = point3d_params;
                keypoint_reprojection_residual->Evaluate(params, residual, NULL);
                double error2 = residual[0] * residual[0] + residual[1] * residual[1];
                if (error2 < 3.0 * sigma) {
                    inliers_mse += error2;
                    ++inliers;
                }
                ++nprojections;
            }
        }
    }
    std::cout << "After BA projections: " << to_percent(inliers, nprojections) << "% inliers with MSE=" << (inliers_mse / inliers) << std::endl;

    std::cout << "After BA ";
    printCamera(camera_intrinsics);

    calib.f_ = camera_intrinsics[2];
    calib.cx_ = camera_intrinsics[3] - 0.5 * calib.width();
    calib.cy_ = camera_intrinsics[4] - 0.5 * calib.height();

    for (size_t camera_id = 0; camera_id < ncameras; ++camera_id) {
        matrix3d R;
        vector3d O;

        phg::decomposeUndistortedPMatrix(R, O, cameras[camera_id]);
        std::cout << "Camera #" << camera_id << " translation: " << O << " -> ";

        double* camera_extrinsics = cameras_extrinsics.data() + CAMERA_EXTRINSICS_NPARAMS * camera_id;
        double* rotation_angle_axis = camera_extrinsics + 0;
        double* translation = camera_extrinsics + 3;

        ceres::AngleAxisToRotationMatrix(rotation_angle_axis, &(R(0, 0)));
        for (int d = 0; d < 3; ++d) {
            O[d] = translation[d];
        }
        std::cout << O << std::endl;

        cameras[camera_id] = phg::composeCameraMatrixRO(R, O);
    }
}

void generateTiePointsCloud(const std::vector<vector3d> tie_points,
                            const std::vector<Track> tracks,
                            const std::vector<std::vector<cv::KeyPoint>> keypoints,
                            const std::vector<cv::Mat> imgs,
                            const std::vector<char> aligned,
                            const std::vector<matrix34d> cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 24152151251241);

    tie_points_colors.clear();
    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const Track &track = tracks[i];
        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    tie_points_and_cameras = tie_points;
    for (int i_camera = 0; i_camera < ncameras; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points_and_cameras.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points_and_cameras.push_back(O + R * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }
}
