#pragma once

#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>

#include "utils/distance.h"

namespace alp {

  template<typename vec_T>
  using data_ptr = vec_T *;


  template<typename vec_t>
  struct Kahan_Average {
      Kahan_Average(size_t dim) : dim_(dim) {
          average_.reserve(dim);
          residual_.resize(dim);
      }

      void add(data_ptr <vec_t> data_ptr) {
          if (count_ == 0) {
              average_.assign(data_ptr, data_ptr + dim_);
              count_++;
              return;
          }

          const vec_t *data = data_ptr;
          vec_t *avg = average_.data();
          vec_t *res = residual_.data();

          for (size_t i = 0; i < dim_; ++i) {
              double delta = (data[i] - avg[i]) / static_cast<double>(count_ + 1);
              vec_t y = delta - res[i];
              vec_t t = avg[i] + y;
              res[i] = (t - avg[i]) - y;
              avg[i] = t;
          }
          count_++;
      }

      void clear() {
          average_.clear();
          residual_.clear();
          count_ = 0;
      }

      size_t count_ = 0;
      size_t dim_;
      std::vector<vec_t> average_;
      std::vector<vec_t> residual_;
  };

  template<typename vec_t>
  struct KMeans {
      KMeans(int k, int max_iters, float tolerance, size_t dim, DistanceType type = L2)
              : k(k), max_iters(max_iters), tolerance(tolerance), distance_calc_(type), dim_(dim) {
      }

      void add(data_ptr<vec_t> data) {
          data_.push_back(data);
      }


      std::vector<std::vector<vec_t>> centroids() {
          return std::move(centroids_datas_);
      }


      void init_centroids() {
          if (data_.empty()) {
              return;
          }

          is_centroid = true;

          centroids_.clear();


          k = std::min(k, static_cast<int>(data_.size()));

          centroids_.assign(data_.begin(), data_ + k);

          std::mt19937 gen(std::random_device{}());

          std::unordered_set<size_t> selected;


          for (auto m = k; m < data_.size(); ++m) {
              std::uniform_int_distribution<> dist(0, m);
              const int j = dist(gen);

              if (j < k) {
                  centroids_[j] = data_[m];
              }
          }
      }

      void clear() {
          centroids_.clear();
          data_.clear();
      }


      void train() {
          if (!is_centroid) {
              init_centroids();
          }
          for (const auto &data: data_) {
              int best_centroid = -1;
              float best_distance = std::numeric_limits<float>::max();

              for (int j = 0; j < k; ++j) {
                  float distance = distance_calc_(data->data(), centroids_[j]->data(), dim_);
                  if (distance < best_distance) {
                      best_distance = distance;
                      best_centroid = j;
                  }
              }

              trained_centroids_[best_centroid].add(data->data());
          }
          bool converged = true;

          for (int j = 0; j < k; ++j) {
              auto &centroid = trained_centroids_[j];
              auto &old_centroid = centroids_[j];

              float distance = distance_calc_(centroid.average_.data(), old_centroid->data(), dim_);

              centroids_datas_[j] = std::move(centroid.average_);
              centroid.clear();
              if (distance > tolerance) {
                  converged = false;
              }
          }

          if (converged) {
              return;
          }

          centroids_.clear();

          for (int i = 1; i < max_iters; ++i) {
              for (const auto &data: data_) {
                  int best_centroid = -1;
                  float best_distance = std::numeric_limits<float>::max();

                  for (int j = 0; j < k; ++j) {
                      float distance = distance_calc_(data->data(), centroids_datas_[j].data(), dim_);
                      if (distance < best_distance) {
                          best_distance = distance;
                          best_centroid = j;
                      }
                  }

                  trained_centroids_[best_centroid].add(data->data());
              }
              converged = true;

              for (int j = 0; j < k; ++j) {
                  auto &centroid = trained_centroids_[j];
                  auto &old_centroid = centroids_datas_[j];

                  float distance = distance_calc_(centroid.average_.data(), old_centroid->data(), dim_);
                  centroids_datas_[j] = std::move(centroid.average_);
                  centroid.clear();

                  if (distance > tolerance) {
                      converged = false;
                  }
              }

              if (converged) {
                  break;
              }
          }
      }

      int k;
      int max_iters;
      float tolerance;

      DistanceCalc<vec_t> distance_calc_;

      bool is_centroid = false;

      std::vector<data_ptr<vec_t>> centroids_;

      std::vector<std::vector<vec_t>> centroids_datas_;

      std::vector<Kahan_Average<vec_t>> trained_centroids_;

      size_t dim_;

      std::vector<data_ptr<vec_t>> data_;
  };

  template<typename vec_t>
  struct KMeansPP {

      KMeansPP(int k, size_t dim, int max_iters = 100, float tolerance = 1e-4, DistanceType type = L2)
              : means_(k, max_iters, tolerance, type) {}
      void centroids_pp(const std::vector<data_ptr<vec_t>> &data) {
          if (data.empty()) {
              return;
          }

          is_pp = true;

          auto &centroids = means_.centroids_;

          centroids.clear();

          std::uniform_int_distribution<> distrib(0, data.size() - 1);

          std::mt19937 gen(std::random_device{}());

          {
              auto dist = distrib(gen);

              centroids.push_back(data[dist]);

              std::unordered_set<size_t> selected;

              selected.emplace(dist);
          }

          size_t k = std::min(static_cast<size_t>(means_.k), data.size());

          auto dim = means_.dim_;

          auto &calc = means_.distance_calc_;

          for (int i = 1; i < k; ++i) {
              std::vector<double> distances(data.size(), 0.0);
              double totalDistance = 0.0;

              for (size_t j = 0; j < data.size(); ++j) {
                  double min_dist = std::numeric_limits<double>::max();
                  for (const auto &c: centroids) {
                      double dist = calc(data[j]->data(), c.data(), dim);
                      min_dist = std::min(min_dist, dist);
                  }
                  distances[j] = min_dist * min_dist;
                  totalDistance += distances[j];
              }

              std::uniform_real_distribution<> prob_dist(0.0, totalDistance);
              double threshold = prob_dist(gen);
              double cumulative = 0.0;
              for (size_t j = 0; j < data.size(); ++j) {
                  cumulative += distances[j];
                  if (cumulative >= threshold) {
                      centroids.push_back(data[j]);
                      break;
                  }
              }
          }

          means_.is_centroid = true;
      }

      std::vector<std::vector<vec_t>> centroids() {
          return means_.centroids();
      }

      void add(data_ptr<vec_t> data) {
          means_.add(data);
      }

      void clear() {
          means_.clear();
      }

      void train() {
          if (is_pp) {
              centroids_pp(means_.data);
          }

          means_.train();
      }

      bool is_pp = false;
      KMeans<vec_t> means_;
  };


}