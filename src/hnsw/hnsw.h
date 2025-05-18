#include <iostream>

#include "ann/index.h"

#include "utils/distance.h"
#include "utils/executor.h"
#include <vector>
#include <cstring>

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cassert>
#include <cstdint>
#include <map>
#include <queue>
#include <set>

namespace alp::hnsw {


  inline std::default_random_engine level_generator_ = std::default_random_engine(0);

  int get_random_level(int mult_) {
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      double r = -log(distribution(level_generator_)) * mult_;
      return (int) r;
  }


  template<typename vec_t>
  class hnsw : public VectorIndex<vec_t> {

  public:
      hnsw(int M, int M_max, int ef_construction, int ef_search) : M_(M), M_max_(M_max),
                                                                   ef_construction_(ef_construction),
                                                                   ef_search_(ef_search),
                                                                   mult_(1 / log(1.0 * M)) {
      }

  private:
      using dis_label_pair = std::pair<float, idx_t>;

      struct Edge {
          [[nodiscard]] int size() const {
              return size_;
          }

          void add_edge(float dis, idx_t label) {
              other_[size_++] = std::make_pair(dis, label);
          }

          void add_edge(std::pair<float, idx_t> &&edge) {
              other_[size_++] = edge;
          }

          void remove_further() {
              // todo change to priority que
              int index = 0;
              auto further = other_[0].first;
              for (int i = 1; i < size_; ++i) {
                  if (other_[i].first > further) {
                      index = i;
                      further = other_[i].first;
                  }
              }
              for (int i = index + 1; i < size_; ++i) {
                  other_[i - 1] = other_[i];
              }
          }

          int remove_further(float dis, idx_t label) {
              // todo change to priority que
              int index = 0;
              auto further = other_[0].first;
              for (int i = 0; i < size_; ++i) {
                  if (other_[i].first > further) {
                      index = i;
                      further = other_[i].first;
                  }
              }
              int llabel = other_[index].second;
              other_[index] = std::make_pair(dis, label);
              return llabel;
          }

          void remove(int label) {
              for (int i = 0; i < size_; ++i) {
                  if (other_[i].second == label) {
                      for (int j = i; j < size_ - 1; ++j) {
                          other_[j] = other_[j + 1];
                      }
                      --size_;
                  }
              }
          }

          int size_{};
          dis_label_pair other_[0];
      };

      // Start at 1, 0 represent the emtpy
      uint32_t max_level_{};

      int entry_label_{};

      Executor scheduler_;

      const int dim{128};

      std::vector<std::map<int, Edge *>> level_edges_;

      std::unordered_map<idx_t , const vec_t *> points_;

      static Edge *create_edge(int M_max_) {
          void *ptr = malloc(sizeof(Edge) + sizeof(dis_label_pair));
          return new(ptr) Edge();
      }

      struct less_cmp {
          bool operator()(const std::pair<float, idx_t> &lhs, const std::pair<float, idx_t> &rhs) const {
              return lhs.first < rhs.first;
          }
      };

      struct greater_cmp {
          bool operator()(const std::pair<float, idx_t> &lhs, const std::pair<float, idx_t> &rhs) const {
              return lhs.first > rhs.first;
          }
      };


      std::set<dis_label_pair, less_cmp>
      search_layer_to_queue(const vec_t *item, idx_t label, uint32_t level, int eq) {
          std::unordered_set<int> visited_set(level_edges_[level - 1].size());
          std::priority_queue<dis_label_pair, std::vector<dis_label_pair>, greater_cmp> wait_que;
          std::set<dis_label_pair, less_cmp> near_neighbor;

          auto cur_point = points_[label];
          auto cur_point_edge = level_edges_[level - 1][label];

          long dis = l2_distance(cur_point, item, dim);

          near_neighbor.emplace(dis, label);
          wait_que.emplace(dis, label);
          while (!wait_que.empty()) {
              wait_que.pop();
              if (dis <= (*near_neighbor.rbegin()).first) [[likely]] {
                  for (int i = 0; i < cur_point_edge->size(); i++) {
                      label = cur_point_edge->other_[i].second;
                      if (visited_set.contains(label)) {
                          continue;
                      }
                      visited_set.insert(label);

                      cur_point = points_[cur_point_edge->other_[i].second];
                      dis = l2distance(item, cur_point, dim);
                      if (dis < (*near_neighbor.rbegin()).first || near_neighbor.size() < eq) {
                          wait_que.emplace(dis, label);

                          if (near_neighbor.size() == eq) {
                              near_neighbor.erase(std::prev(near_neighbor.end()));
                          }
                          near_neighbor.emplace(dis, label);
                      }
                  }
                  dis = wait_que.top().first;
                  cur_point_edge = level_edges_[level - 1][wait_que.top().second];
                  continue;
              }
              break;
          }
          return std::move(near_neighbor);
      }

      int search_layer_down(const vec_t *item, idx_t label, uint32_t level) {
          std::unordered_set<int> visited_set(level_edges_[level - 1].size());
          dis_label_pair near_que;
          std::priority_queue<dis_label_pair, std::vector<dis_label_pair>, greater_cmp> wait_que;

          auto cur_point = points_[label];
          auto cur_point_edge = level_edges_[level - 1][label];

          long dis = l2distance(cur_point, item, dim);

          near_que = std::make_pair(dis, label);
          wait_que.emplace(dis, label);
          while (!wait_que.empty()) {
              wait_que.pop();
              if (dis <= near_que.first) [[likely]] {
                  for (int i = 0; i < cur_point_edge->size(); i++) {
                      label = cur_point_edge->other_[i].second;
                      if (visited_set.contains(label)) {
                          continue;
                      }
                      visited_set.insert(label);

                      cur_point = points_[cur_point_edge->other_[i].second];
                      dis = l2distance(item, cur_point, dim);

                      if (dis < near_que.first) {
                          wait_que.emplace(dis, label);
                          near_que = std::make_pair(dis, label);
                      }
                  }
                  dis = wait_que.top().first;
                  cur_point_edge = level_edges_[level - 1][wait_que.top().second];
                  continue;
              }
              break;
          }
          return near_que.second;
      }

      void search_layer(const vec_t *item, idx_t label, uint32_t level, const int eq, std::vector<idx_t> &n_set) {
          std::unordered_set<int> visited_set(level_edges_[level - 1].size());
          std::priority_queue<dis_label_pair, std::vector<dis_label_pair>, less_cmp> near_que;
          std::priority_queue<dis_label_pair, std::vector<dis_label_pair>, less_cmp> wait_que;

          auto cur_point = points_[label];
          auto cur_point_edge = level_edges_[level - 1][label];

          long dis = l2distance(cur_point, item, dim);

          near_que.emplace(dis, label);
          wait_que.emplace(dis, label);
          while (!wait_que.empty()) {
              if (near_que.empty() || dis > near_que.top().first) [[likely]] {
                  for (int i = 0; i < cur_point_edge->size(); i++) {
                      label = cur_point_edge->other_[i].second;
                      if (visited_set.contains(label)) {
                          continue;
                      }
                      visited_set.insert(label);

                      cur_point = points_[cur_point_edge->other_[i].second];
                      dis = l2distance(item, cur_point, dim);

                      if (dis < near_que.top().first || near_que.size() < eq) {
                          if (near_que.size() == eq) {
                              near_que.pop();
                              near_que.emplace(dis, label);
                          }
                          wait_que.emplace(dis, label);
                          near_que.emplace(dis, label);
                      }
                  }
                  dis = wait_que.top().first;
                  cur_point = points_[wait_que.top().second];
                  wait_que.pop();
                  continue;
              }
              break;
          }
          while (!near_que.empty()) {
              n_set.emplace_back(near_que.top().second);
              near_que.pop();
          }
      }

  public:
      // you can add more parameter to initialize hnsw
      hnsw() = default;

      void insert(const vec_t *item, idx_t label) override;

      std::vector<int> query(const vec_t *query, int k) override;

      void query(const vec_t *query, int k, std::vector<idx_t> *result);

      ~hnsw() override {
          for (auto &level_edge: level_edges_) {
              for (auto &edge: level_edge) {
                  free(edge.second);
              }
          }
      }

  private:
      int M_ = 30;
      int M_max_ = 30;
      int ef_construction_ = 100;
      int ef_search_ = 100;
      const double mult_;
  };

  template<typename vec_t>
  void hnsw<vec_t>::insert(const vec_t *item, idx_t label) {
      uint32_t random_level = get_random_level(mult_) + 1;
      auto level_index = max_level_;
      auto entry_label = entry_label_;
      for (; level_index > random_level; level_index--) {
          entry_label = search_layer_down(item, entry_label, level_index);
      }

      if (random_level > max_level_) {
          for (auto i = max_level_; i < random_level; i++) {
              level_edges_.emplace_back();
              level_edges_[i].emplace(label, create_edge());
          }
      }

      std::set<dis_label_pair, less_cmp> que;
      dis_label_pair neighbour_pair;
      Edge *neighbour_edge;
      Edge *cur_level_edge;
      points_.emplace(label, item);
      for (; level_index > 0; level_index--) {
          que = std::move(search_layer_to_queue(item, entry_label, level_index, ef_construction_));
          entry_label = (*que.begin()).second;
          cur_level_edge = create_edge();

          level_edges_[level_index - 1].emplace(label, cur_level_edge);

          int m_neighbour = 0;
          for (auto it = que.begin(); it != que.end() && m_neighbour < M_; it++, ++m_neighbour) {
              neighbour_pair = (*it);
              neighbour_edge = level_edges_[level_index - 1][neighbour_pair.second];
              cur_level_edge->add_edge(neighbour_pair.first, neighbour_pair.second);
              if (neighbour_edge->size() == M_max_) {
                  int rm_label = neighbour_edge->remove_further(neighbour_pair.first, label);
                  neighbour_edge = level_edges_[level_index - 1][rm_label];
                  neighbour_edge->remove(neighbour_pair.second);
              } else {
                  neighbour_edge->add_edge(neighbour_pair.first, label);
              }
          }

      }

      if (random_level > max_level_) {
          max_level_ = random_level;
          entry_label_ = label;
      }

  }

  template<typename vec_t>
  std::vector<int> hnsw<vec_t>::query(const vec_t *query, int k) {
      std::vector<int> res;
      res.reserve(k);
      if (max_level_) {
          int entry_label = entry_label_;
          for (auto level = max_level_; level > 1; level--) {
              entry_label = search_layer_down(query, entry_label, level);
          }
          auto que = std::move(search_layer_to_queue(query, entry_label, 1, ef_construction_));
          for (auto it = que.begin(); it != que.end() && k > 0; it++, k--) {
              res.emplace_back((*it).second);
          }
      }
      return std::move(res);
  }

  template<typename vec_t>
  void hnsw<vec_t>::query(const vec_t *query, int k, std::vector<idx_t> *res) {
      res->reserve(k);
      if (max_level_) {
          int entry_label = entry_label_;
          for (auto level = max_level_; level > 1; level--) {
              entry_label = search_layer_down(query, entry_label, level);
          }
          auto que = std::move(search_layer_to_queue(query, entry_label, 1, ef_construction_));
          for (auto it = que.begin(); it != que.end() && k > 0; it++, k--) {
              res->emplace_back((*it).second);
          }
      }
  }

}