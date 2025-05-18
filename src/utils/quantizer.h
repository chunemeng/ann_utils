#pragma once

#include <vector>
#include "utils/distance.h"
#include <cassert>
#include <algorithm>
#include "utils/kmeans.h"


namespace alp {

  template<typename vec_t>
  struct Minmax {
      vec_t min_val{std::numeric_limits<vec_t>::max()};
      vec_t max_val{std::numeric_limits<vec_t>::min()};
  };

  template<typename T, typename vec_t>
  static constexpr inline T clamp2T(vec_t val, const Minmax<vec_t> &minmax, double diff) {
      return static_cast<T>(2.0 * ((static_cast<double>(val) - minmax.min_val) /
                                   diff - 0.5) * std::numeric_limits<T>::max());
  }

  template<typename T, typename vec_t>
  static constexpr inline T clampT2(T val, const Minmax<vec_t> &minmax, double diff) {
      return (static_cast<double>(val) / 2.0 / static_cast<double >(std::numeric_limits<T>::max()) + 0.5) *
             diff + minmax.min_val;
  }


  template<typename T, typename vec_t>
  static constexpr inline T clamp2T(vec_t val, const Minmax<vec_t> &minmax) {
      return clamp2T<T, vec_t>(val, minmax, (static_cast<double >(minmax.max_val) - minmax.min_val));
  }

  template<typename T, typename vec_t>
  static constexpr inline T clampT2(T val, const Minmax<vec_t> &minmax) {
      return clampT2<T, vec_t>(val, minmax, (static_cast<double >(minmax.max_val) - minmax.min_val));
  }


  template<typename T, typename vec_t>
  static inline std::vector<T> scalar_quantize(const Minmax<vec_t> &minmax, const vec_t *data, size_t dim,
                                               double diff) {
      std::vector<T> result;
      result.reserve(dim);
      for (size_t i = 0; i < dim; ++i) {
          result.push_back(clamp2T(data[i], minmax, diff));
      }
      return result;
  }

  template<typename T, typename vec_t>
  static inline std::vector<vec_t> scalar_dequantize(const Minmax<vec_t> &minmax, const T *data, size_t dim,
                                                     double diff) {
      std::vector<vec_t> result;
      result.reserve(dim);
      for (size_t i = 0; i < dim; ++i) {
          result.push_back(clampT2(data[i], minmax, diff));
      }
      return result;
  }

  template<typename T, typename vec_t>
  static inline std::vector<vec_t> scalar_dequantize_with_plus(const Minmax<vec_t> &minmax, const T *data, size_t dim,
                                                               const T *bias,
                                                               double diff) {
      std::vector<vec_t> result;
      result.reserve(dim);
      for (size_t i = 0; i < dim; ++i) {
          result.push_back(clampT2(data[i], minmax, diff) + bias[i]);
      }
      return result;
  }


  template<typename T, typename vec_t>
  class IVF_ScalarQuantizer {
  private:
      std::vector<vec_t> pre_handle(const vec_t *data, size_t dim) {
          std::vector<vec_t> result;
          result.reserve(dim);
          for (size_t i = 0; i < dim; ++i) {
              result.emplace_back(data[i] - cluster_centers_[i]);
          }
          return result;
      }

      std::vector<vec_t> de_pre_dequantize(const vec_t *data, size_t dim) {
          scalar_dequantize_with_plus(minmax_, data, dim, cluster_centers_.data(), diff_);
      }

  public:
      IVF_ScalarQuantizer(std::vector<vec_t> &cluster_centers)
              : cluster_centers_(cluster_centers) {
      }

      IVF_ScalarQuantizer() = delete;

      void clear() {
          clusters_.clear();
      }

      std::vector<std::vector<T>> train_clusters() {
          std::vector<std::vector<T>> quantized_clusters_;
          auto dim = cluster_centers_.size();
          quantized_clusters_.reserve(clusters_.size());
          for (size_t i = 0; i < clusters_.size(); ++i) {
              quantized_clusters_.emplace_back(quantize_cluster(clusters_[i].data(), dim));
          }
          diff_ = (static_cast<double >(minmax_.max_val) - minmax_.min_val);
          scale_ = diff_ / static_cast<double >(std::numeric_limits<T>::max()) / 2.0;

          return quantized_clusters_;
      }


      void add_cluster(const vec_t *data, size_t dim) {
          clusters_.emplace_back(pre_handle(data, dim));
          update_minmax(clusters_.back().data(), dim);
      }

      std::vector<T> quantize_cluster(const vec_t *data, size_t dim) {
          return scalar_quantize(minmax_, data, dim, diff_);
      }

      std::vector<vec_t> dequantize_cluster(const T *data, size_t dim) {
          return scalar_dequantize(minmax_, data, dim, diff_);
      }

      float compute_distance_l2(const vec_t *qvec, const T *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = dequantize_cluster(c_vec, dim_);

          return l2_distance<T>(qvec, dequantized_vec.data(), dim_);
      }

      float compute_distance_l2(const T *qvec, const T *c_vec) const {
          return l2_distance<vec_t>(qvec, c_vec, cluster_centers_.size()) * (scale_ * scale_);
      }

      float compute_distance_ip(const vec_t *qvec, const T *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = de_pre_dequantize(c_vec, dim_);

          return ip_distance<vec_t>(qvec, dequantized_vec.data(), dim_);
      }

      float compute_distance_ip(const T *qvec, const T *c_vec) const {
          assert(false && "Not support");
      }

      float compute_distance_cosine(const vec_t *qvec, const T *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = de_pre_dequantize(c_vec, dim_);

          return cosine_distance<vec_t>(qvec, dequantized_vec.data(), dim_);
      }


      float compute_distance_cosine(const T *qvec, const T *c_vec) const {
          assert(false && "Not support");
      }

  private:

      void update_minmax(const vec_t *data, size_t size) const {
          for (size_t i = 0; i < size; ++i) {
              minmax_.min_val = std::min(minmax_.min_val, data[i]);
              minmax_.max_val = std::max(minmax_.max_val, data[i]);
          }
      }

      double diff_;
      double scale_;

      std::vector<std::vector<vec_t>> clusters_;
      std::vector<vec_t> &cluster_centers_;
      Minmax<vec_t> minmax_;
  };


  template<typename vec_t>
  class IVF_ProductQuantizer {
  private:
      std::vector<vec_t> pre_handle(const vec_t *data, size_t dim) {
          std::vector<vec_t> result;
          result.reserve(dim);
          for (size_t i = 0; i < dim; ++i) {
              result.emplace_back(data[i] - cluster_centers_[i]);
          }
          return result;
      }

  public:
      IVF_ProductQuantizer(std::vector<std::vector<vec_t>> &cluster_centers, int m = 8)
              : cluster_centers_(cluster_centers), m_(m) {
          assert(m > 0);
          chunk_size_ = (cluster_centers_.size()) / m_;
          codebooks_.reserve(m_);
      }

      void clear() {
          clusters_.clear();
      }

      std::vector<std::vector<uint8_t>> train_clusters() {
          std::vector<std::vector<uint8_t>> quantized_clusters_;
          if (chunk_size_ == 0) {
              return quantized_clusters_;
          }

          assert(clusters_.size() > 8);
          auto dim = cluster_centers_.size();

          auto last_dim = dim - (m_ - 1) * chunk_size_;

          std::vector<KMeansPP<vec_t>> kmeans;
          kmeans.reserve(m_);

          for (size_t i = 0; i < m_; ++i) {
              if (i == m_ - 1) {
                  kmeans.emplace_back(last_dim, dim);
              } else {
                  kmeans.emplace_back(chunk_size_, dim);
              }
          }

          for (const auto &data: clusters_) {
              for (size_t i = 0; i < m_; ++i) {
                  kmeans[i].add(data.data() + i * chunk_size_);
              }
          }

          for (size_t i = 0; i < m_; ++i) {
              kmeans[i].train();
              codebooks_.emplace_back(kmeans[i].centroids());
          }

          quantized_clusters_.reserve(clusters_.size());
          for (const auto &data: clusters_) {
              quantized_clusters_.emplace_back(quantize_cluster(data.data(), dim));
          }

          for (auto i = 0; i < m_; ++i) {
              auto &codebook = codebooks_[i];
              auto &cluster = clusters_[i];
              for (size_t j = 0; j < codebook.size(); ++j) {
                  auto &code = codebook[j];
                  for (size_t k = 0; k < code.size(); ++k) {
                      code[k] += cluster_centers_[i * chunk_size_ + k];
                  }
              }
          }

          return quantized_clusters_;
      }


      void add_cluster(const vec_t *data, size_t dim) {
          clusters_.emplace_back(pre_handle(data, dim));
      }

      std::vector<uint8_t> quantize_cluster(const vec_t *data, size_t dim) {
          std::vector<uint8_t> result;
          result.reserve(m_);

          for (size_t i = 0; i < m_; ++i) {
              auto &codebook = codebooks_[i];
              auto data_chunk = data + i * chunk_size_;
              float min_dis = std::numeric_limits<float>::max();
              int min_index = -1;


              for (size_t k = 0; k < codebook.size(); ++k) {
                  auto dis = l2_distance(data_chunk, codebook[k].data(), codebook[k].size());
                  if (dis < min_dis) {
                      min_dis = dis;
                      min_index = static_cast<int>(k);
                  }
              }
              result.push_back(static_cast<uint8_t>(min_index));
          }


          return result;
      }

      std::vector<vec_t> dequantize_cluster(const uint8_t *data, size_t dim) {
          std::vector<vec_t> result;
          result.reserve(dim);

          for (size_t i = 0; i < m_; ++i) {
              auto &codebook = codebooks_[i];
              auto index = data[i];
              result.insert(result.end(), codebook[index].begin(), codebook[index].end());
          }

          return result;
      }

      float compute_distance_l2(const vec_t *qvec, const uint8_t *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = dequantize_cluster(c_vec, dim_);

          return l2_distance<vec_t>(qvec, dequantized_vec.data(), dim_);
      }

      float compute_distance_ip(const vec_t *qvec, const uint8_t *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = dequantize_cluster(c_vec, dim_);

          return ip_distance<vec_t>(qvec, dequantized_vec.data(), dim_);
      }

      float compute_distance_cosine(const vec_t *qvec, const uint8_t *c_vec) const {
          const size_t dim_ = cluster_centers_.size();

          auto dequantized_vec = dequantize_cluster(c_vec, dim_);

          return cosine_distance<vec_t>(qvec, dequantized_vec.data(), dim_);
      }

  private:
      std::vector<std::vector<vec_t>> clusters_;
      std::vector<vec_t> &cluster_centers_;

      std::vector<std::vector<std::vector<vec_t>>> codebooks_;
      size_t chunk_size_;
      int m_ = 8;
  };

}
