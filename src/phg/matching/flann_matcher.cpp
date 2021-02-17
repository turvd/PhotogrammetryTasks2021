#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // TOOO remove numbers
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    const int n_queries = query_desc.rows;

    // сделать запрос к построенному индексу
    cv::Mat indices(n_queries, k, CV_32SC1);
    cv::Mat dists2(n_queries, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, dists2, k, *search_params);

    // сконвертировать в объекты типа DMatch
    matches.resize(n_queries);
    for (int i = 0; i < n_queries; ++i) {
        matches[i].clear();
        matches[i].reserve(k);
        for (int ki = 0; ki < k; ++ki) {
            cv::DMatch match;
            match.imgIdx = 0;
            match.queryIdx = i;
            match.trainIdx = indices.at<int>(i, ki);
            match.distance = std::sqrt(dists2.at<float>(i, ki));
            matches[i].push_back(match);
        }
        if (matches[i][0].distance > matches[i][1].distance) {
            std::cerr << "ooops" << std::endl;
        }
    }
}

