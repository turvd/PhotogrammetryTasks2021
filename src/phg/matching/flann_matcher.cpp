#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices;
    cv::Mat distances2;
    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);
    for (int i = 0; i < indices.rows; ++i) {
        std::vector<cv::DMatch> knn_matches;
        for (int j = 0; j < k; ++j) {
            knn_matches.push_back(cv::DMatch(i, indices.at<int>(i, j), sqrt(distances2.at<float>(i, j))));
        }
        matches.push_back(knn_matches);
    }
    //throw std::runtime_error("not implemented yet");
}
