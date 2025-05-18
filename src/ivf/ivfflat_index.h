#pragma once

#include "ann/index.h"
#include "utils/distance.h"
#include "storage/storage.h"
#include "utils/quantizer.h"
#include "utils/bounded_priority_queue.h"
#include "utils/kmeans.h"
#include <cassert>
#include <stdfloat>
#include <unordered_map>


namespace alp::ivf {


  using idx_t = int64_t;

  enum ClusterType {
      kFlat = 0,
      kPQ = 1,
      kSQ_INT8 = 2,
      kSQ_FP16 = 3,
      kSQ_FP32 = 4,
      kSQ_BF16 = 5,
  };


  struct IvfIndexFileHeader {
      int lists_ = 0;
      int probes_ = 0;
      int dim_ = 0;
      int distance_type_ = 0;
      ClusterType cluster_type_ = kFlat;
  };


  struct predict_result {
      idx_t id;
      float dis;
  };

  template<typename T>
  struct data_type {
      data_type(const T *begin, const T *end, idx_t id) : id(id) {
          data.reserve(end - begin);
          data.insert(data.end(), begin, end);
      }

      data_type(std::vector<T> &&d, idx_t id) : data(std::move(d)), id(id) {}

      std::vector<T> data;
      idx_t id;
  };


  template<typename vec_t>
  struct IvfCluster {
      using idx_t = int64_t;

      using predict_type = bounded_priority_queue<predict_result, std::greater<>>;

      struct ClusterData {
          virtual size_t data_num() const = 0;

          virtual ClusterType type() const = 0;

          const std::vector<vec_t> &centroid() const {
              return centroid_;
          }

          virtual void add(const vec_t *vec_ptr, idx_t id, size_t dim) = 0;

          virtual predict_type predict(int k, const vec_t *vec_ptr, size_t dim,
                                       DistanceType type = L2) = 0;

          virtual ~ClusterData() = default;

          std::vector<vec_t> centroid_;
      };

      template<typename T>
      struct ClusterDataT {
          size_t data_num() const {
              return datas_.size();
          }

          void add(const vec_t *vec_ptr, idx_t id, size_t dim) {
              datas_.emplace_back(vec_ptr, vec_ptr + dim, id);
          }

          void add(std::vector<T> &&d, idx_t id) {
              datas_.emplace_back(std::move(d), id);
          }

          predict_type predict(int k, const vec_t *vec_ptr, size_t dim,
                               DistanceType type) {
              bounded_priority_queue<predict_result, std::greater<>> queue(k);

              DistanceCalc<vec_t> calc(type);

              for (const auto &data: datas_) {
                  auto dis = calc(vec_ptr, data.data.data(), dim);
                  queue.push({data.id, dis});
              }
              return queue;
          }

          void reserve(size_t size) {
              datas_.reserve(size);
          }

          void clear() {
              datas_.clear();
          }

          std::vector<data_type<T>> datas_;
      };

      struct FlatData : public ClusterData {
          FlatData(std::vector<vec_t> &&cent) : ClusterData(std::move(cent)) {
          }

          size_t data_num() const override {
              return data_.data_num();
          }

          ClusterType type() const override {
              return kFlat;
          }

          void add(const vec_t *vec_ptr, idx_t id, size_t dim) override {
              data_.add(vec_ptr, id, dim);
          }

          predict_type predict(int k, const vec_t *vec_ptr, size_t dim,
                               DistanceType type) override {
              return data_.predict(k, vec_ptr, dim, type);
          }

          void reserve(size_t size) override {
              data_.reserve(size);
          }

          void clear() override {
              data_.clear();
          }

          ClusterDataT<vec_t> data_;
      };

      template<typename T>
      struct SQData : public ClusterData {
          size_t data_num() const override {
              return sq_data_.data_num();
          }

          SQData(std::vector<T> &&cent) : ClusterData(std::move(cent)), quantizer_(ClusterData::centroid()) {
          }

          ClusterType type() const override {
              if constexpr (std::is_same_v<T, int8_t>()) {
                  return kSQ_INT8;
              } else if constexpr (std::is_same_v<T, float>()) {
                  return kSQ_FP32;
              } else if constexpr (std::is_same_v<T, std::float16_t>()) {
                  return kSQ_FP16;
                  #if defined(__GNUC__) && (__GNUC__ > 13) && defined(__STDCPP_BFLOAT16_T__)
                  } else if constexpr (std::is_same_v<T, std::bfloat16_t>()) {
                      return kSQ_BF16;
                  #endif
              }
              return kSQ_FP32;
          }


          void add(const vec_t *vec_ptr, idx_t id, size_t dim) override {
              data_.add(vec_ptr, id, dim);
              quantizer_.add(vec_ptr, dim);
          }

          void reserve(size_t size) {
              data_.reserve(size);
              sq_data_.reserve(size);
          }

          void train() {
              sq_data_.clear();
              auto sq_d = quantizer_.train_clusters();
              for (size_t i = 0; i < sq_d.size(); ++i) {
                  sq_data_.add(std::move(sq_d), data_[i].id);
              }
              data_.clear();
          }

          predict_type predict(int k, const vec_t *vec_ptr, size_t dim, DistanceType type) override {
              bounded_priority_queue<predict_result, std::greater<>> queue(k);
              for (const auto &data: sq_data_.datas_) {
                  float dis = 0;
                  switch (type) {
                      case L2:
                          dis = quantizer_.compute_distance_l2(vec_ptr, data.data.data());
                          break;
                      case IP:
                          dis = quantizer_.compute_distance_ip(vec_ptr, quantizer_.dequantize_cluster(data.data.data(),
                                                                                                      data.data.size()));
                          break;
                      case COSINE:
                          dis = quantizer_.compute_distance_cosine(vec_ptr,
                                                                   quantizer_.dequantize_cluster(data.data.data(),
                                                                                                 data.data.size()));
                          break;

                      case UNKNOWN:
                          assert(false);
                          break;
                  }
                  queue.push({data.id, dis});
              }
              return queue;
          }

          IVF_ScalarQuantizer<T, vec_t> quantizer_;
          ClusterDataT<vec_t> data_;
          ClusterDataT<T> sq_data_;
      };

      std::unique_ptr<ClusterData> add_cluster(std::vector<vec_t> &&centroid, ClusterType type) {
          std::unique_ptr<ClusterData> ptr;
          switch (type) {
              case kFlat:
                  ptr = std::make_unique<FlatData>(std::move(centroid));
                  break;
              case kSQ_INT8:
                  ptr = std::make_unique<SQData<int8_t>>(std::move(centroid));
                  break;
              case kSQ_FP16:
                  ptr = std::make_unique<SQData<std::float16_t>>(std::move(centroid));
                  break;
              case kSQ_FP32:
                  ptr = std::make_unique<SQData<float>>(std::move(centroid));
                  break;
              case kPQ:
                  ptr = std::make_unique<PQData>(std::move(centroid));
                  break;

              default:
                  assert(false);
                  ptr = std::make_unique<ClusterDataT<vec_t>>();
                  break;
          }

          return datas_.emplace_back(std::move(ptr));
      }


      struct PQData : public ClusterData {
          size_t data_num() const override {
              return datas_.size();
          }

          PQData(std::vector<vec_t> &&cent) : ClusterData(std::move(cent)) {
          }

          ClusterType type() const override {
              return kPQ;
          }

          void add(const vec_t *vec_ptr, idx_t id, size_t dim) override {
              data_.add(vec_ptr, id, dim);
              quantizer_.add(vec_ptr, dim);
          }

          void reserve(size_t size) {
              data_.reserve(size);
              pq_data_.reserve(size);
          }

          void train() {
              pq_data_.clear();
              auto sq_d = quantizer_.train_clusters();
              for (size_t i = 0; i < sq_d.size(); ++i) {
                  pq_data_.add(std::move(sq_d), data_[i].id);
              }
              data_.clear();
          }

          predict_type predict(int k, const vec_t *vec_ptr, size_t dim, DistanceType type) override {
              bounded_priority_queue<predict_result, std::greater<>> queue(k);
              for (const auto &data: pq_data_.datas_) {
                  float dis = 0;
                  switch (type) {
                      case L2:
                          dis = quantizer_.compute_distance_l2(vec_ptr, data.data.data());
                          break;
                      case IP:
                          dis = quantizer_.compute_distance_ip(vec_ptr, data.data.data());
                          break;
                      case COSINE:
                          dis = quantizer_.compute_distance_cosine(vec_ptr,
                                                                   data.data.data());
                          break;
                      case UNKNOWN:
                          assert(false);
                          break;
                  }
                  queue.push({data.id, dis});
              }
              return queue;
          }

          IVF_ProductQuantizer<vec_t> quantizer_;
          ClusterDataT<vec_t> data_;
          ClusterDataT<uint8_t> pq_data_;
      };

      size_t size() const {
          return datas_.size();
      }

      std::unique_ptr<ClusterData> &get_cluster(int i) {
          return datas_[i];
      }

      std::unique_ptr<ClusterData> &operator[](int i) {
          return datas_[i];
      }

      void reserve(size_t size) {
          datas_.reserve(size);
      }

      std::vector<std::unique_ptr<ClusterData>> datas_;
  };

  template<typename vec_t = float>
  class IvfIndex : public alp::VectorIndex<vec_t> {
  public:
      IvfIndex(ClusterType c_type, int lists, int probes, int dim, DistanceType type);

      ~IvfIndex() noexcept = default;

      Status add(idx_t id, const vec_t *vec_ptr) override;

      Status build() override;

      Status search(const vec_t *query_vec, size_t k,
                    std::vector<idx_t> &result_ids,
                    std::vector<float> &result_distances) const override;

      size_t dimension() const override;

      size_t size() const override;


  private:
      bool is_inited_ = false;
      IvfIndexFileHeader header_;

      IvfCluster<vec_t> ivf_clusters_;

      std::unordered_map<idx_t, data_type<vec_t>> datas_;
      DistanceCalc<vec_t> calc_;

      KMeansPP<vec_t> kmeans_;
  };

}