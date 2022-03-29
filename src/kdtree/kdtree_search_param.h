#pragma once

namespace kdtree {

class KDTreeSearchParams {
public:
    enum class SearchType {
        KNN = 0,
        Radius = 1,
    };
    KDTreeSearchParams(float radius, int max_nn)
        : search_type_(SearchType::Radius),
          radius_(radius),
          max_nn_(max_nn) {}
    KDTreeSearchParams(int knn = 30)
            : search_type_(SearchType::KNN),
              knn_(knn) {}
public:
    SearchType search_type_;
    float radius_;
    int max_nn_;
    int knn_;
};

}  // namespace kdtree
