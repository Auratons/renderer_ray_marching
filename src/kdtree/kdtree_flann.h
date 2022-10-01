#pragma once

#define ERROR_MSG(x) std::cout << x << std::endl;

#include <memory>
#include <tuple>
//#include <torch/torch.h>
#include <thrust/device_vector.h>

#include "flann/flann.hpp"
#include "kdtree/kdtree_search_param.h"

namespace kdtree {

class KDTreeFlann {
public:
    KDTreeFlann() = default;
    ~KDTreeFlann() = default;
    KDTreeFlann(const KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

    bool Build(const thrust::device_vector<float> &pcd);
    bool Build(float *pcd, size_t pcd_size);
    int Search(const thrust::device_vector<float> &query,
               const KDTreeSearchParams &param,
               thrust::device_vector<int> &indices,
               thrust::device_vector<float> &distance2) const;
//    std::tuple<thrust::device_vector<float>, thrust::device_vector<float>> SearchTorch(const thrust::device_vector<float> &query,
//                                                         const KDTreeSearchParams &param) const;

    int SearchKNN(const thrust::device_vector<float> &query,
                  int knn,
                  thrust::device_vector<int> &indices,
                  thrust::device_vector<float> &distance2) const;
    int SearchRadius(const thrust::device_vector<float> &query,
                     float radius,
                     int max_nn,
                     thrust::device_vector<int> &indices,
                     thrust::device_vector<float> &distance2) const;

    std::unique_ptr<flann::Matrix<float>> flann_dataset_;
    std::unique_ptr<flann::KDTreeCuda3dIndex<flann::L2<float>>> flann_index_;
};

}  // namespace kdtree
