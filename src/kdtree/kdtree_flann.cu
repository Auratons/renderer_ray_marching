#include "kdtree/kdtree_flann.h"

namespace kdtree {

int KDTreeFlann::Search(thrust::device_vector<float> &query,
                        const KDTreeSearchParams &param,
                        thrust::device_vector<int> &indices,
                        thrust::device_vector<float> &distance2) const {
//    if (!query.device().is_cuda() || pcd_.device().index() != query.device().index() || !query.is_contiguous()
//        || !(query.dtype() == torch::kFloat32)) {
//        ERROR_MSG("[KDTreeFlann::Search] tensor must be float32, contiguous and on the correct cuda device!")
//        return -1;
//    }
//    if (query.dim() != 2 || query.size(1) != 4 || query.size(0) == 0) {
//        ERROR_MSG("[KDTreeFlann::Search] tensor must be of shape [n, 4] and nonempty!")
//        return -1;
//    }
    // cudaSetDevice(pcd_.device().index());

    switch (param.search_type_) {
        case KDTreeSearchParams::SearchType::KNN:
            return SearchKNN(query, param.knn_, indices, distance2);
        case KDTreeSearchParams::SearchType::Radius:
            return SearchRadius(query, param.radius_, param.max_nn_, indices, distance2);
        default:
            return -1;
    }
}

//std::tuple<torch::Tensor, torch::Tensor> KDTreeFlann::SearchTorch(const torch::Tensor &query,
//                                                                  const KDTreeSearchParams &param) const {
//    if (!query.device().is_cuda() || pcd_.device().index() != query.device().index() || !query.is_contiguous()
//        || !(query.dtype() == torch::kFloat32)) {
//        ERROR_MSG("[KDTreeFlann::Search] tensor must be float32, contiguous and on the correct cuda device!")
//        return std::make_tuple(torch::empty({0, 0}), torch::empty({0, 0}));;
//    }
//    if (query.dim() != 2 || query.size(1) != 4 || query.size(0) == 0) {
//        ERROR_MSG("[KDTreeFlann::Search] tensor must be of shape [n, 4] and nonempty!")
//        return std::make_tuple(torch::empty({0, 0}), torch::empty({0, 0}));;
//    }
//    //cudaSetDevice(pcd_.device().index());
//    thrust::device_vector<int> indices;
//    thrust::device_vector<float> distance2;
//
//    int retval = -1;
//    switch (param.search_type_) {
//        case KDTreeSearchParams::SearchType::KNN:
//            retval = SearchKNN(query, param.knn_, indices, distance2);
//        case KDTreeSearchParams::SearchType::Radius:
//            retval = SearchRadius(query, param.radius_, param.max_nn_, indices, distance2);
//    }
//    if (retval == -1) {
//        return std::make_tuple(torch::empty({0, 0}), torch::empty({0, 0}));
//    }
//
//    torch::Tensor I = torch::from_blob(thrust::raw_pointer_cast(indices.data()),
//                                       {query.size(0), (long) indices.size() / query.size(0)},
//                                       torch::TensorOptions().dtype(torch::kInt32).device(query.device())).clone();
//    torch::Tensor D = torch::from_blob(thrust::raw_pointer_cast(distance2.data()),
//                                       {query.size(0), (long) distance2.size() / query.size(0)},
//                                       torch::TensorOptions().dtype(torch::kFloat32).device(query.device())).clone();
//    return std::make_tuple(I, D);
//}

int KDTreeFlann::SearchKNN(thrust::device_vector<float> &query,
                           int knn,
                           thrust::device_vector<int> &indices,
                           thrust::device_vector<float> &distance2) const {
    const int64_t n_query = query.size() / 4;
    flann::Matrix<float> query_flann(thrust::raw_pointer_cast(query.data()), n_query, 3, sizeof(float) * 4);

    indices.resize(n_query * knn);
    distance2.resize(n_query * knn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), n_query, knn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), n_query, knn);

    flann::SearchParams param;
    param.matrices_in_gpu_ram = true;
    return flann_index_->knnSearch(query_flann, indices_flann, dists_flann, knn, param);
}

int KDTreeFlann::SearchRadius(thrust::device_vector<float> &query,
                              float radius,
                              int max_nn,
                              thrust::device_vector<int> &indices,
                              thrust::device_vector<float> &distance2) const {
    const int64_t n_query = query.size() / 4;
    flann::Matrix<float> query_flann(thrust::raw_pointer_cast(query.data()), n_query, 3, sizeof(float) * 4);

    indices.resize(n_query * max_nn);
    distance2.resize(n_query * max_nn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), n_query, max_nn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), n_query, max_nn);

    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    param.matrices_in_gpu_ram = true;
    return flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,float(radius * radius), param);
}

bool KDTreeFlann::Build(thrust::device_vector<float> &pcd) {
//    if (!pcd.device().is_cuda() || !pcd.is_contiguous() || !(pcd.dtype() == torch::kFloat32)) {
//        ERROR_MSG("[KDTreeFlann::Build] tensor must be float32, contiguous and on a cuda device!")
//        return false;
//    }
//    if (pcd.dim() != 2 || pcd.size(1) != 4 || pcd.size(0) == 0) {
//        ERROR_MSG("[KDTreeFlann::Build] tensor must be of shape [n, 4] and nonempty!")
//        return false;
//    }
    // cudaSetDevice(pcd.device().index());

    flann_dataset_ = std::make_unique<flann::Matrix<float>>(thrust::raw_pointer_cast(pcd.data()), pcd.size() / 4, 3, sizeof(float) * 4);
    flann::KDTreeCuda3dIndexParams index_params;
    flann_index_ = std::make_unique<flann::KDTreeCuda3dIndex<flann::L2<float>>>(*flann_dataset_, index_params);
    flann_index_->buildIndex();
    pcd_ = pcd;
    return true;
}

}  // namespace kdtree
