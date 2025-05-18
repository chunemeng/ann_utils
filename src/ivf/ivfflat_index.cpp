#include "ivfflat_index.h"

namespace alp::ivf {
  template<typename vec_t>
  Status IvfIndex<vec_t>::build() {
      if (is_inited_) {
          return Status::OK();
      }

      is_inited_ = true;

      kmeans_.train();

      auto ce = kmeans_.centroids();
      auto cluster_num = ce.size();
      auto dim = header_.dim_;

      kmeans_.clear();

      {
          auto ce_cp = ce;
          ivf_clusters_.reserve(ce_cp.size());
          for (size_t i = 0; i < ce_cp.size(); ++i) {
              ivf_clusters_.add_cluster(std::move(ce_cp), header_.cluster_type_);
          }
      }


      for (const auto &data: datas_) {
          float min_dis = std::numeric_limits<float>::max();
          int min_index = -1;
          for (size_t i = 0; i < cluster_num; ++i) {
              auto &c = ce[i];
              auto dis = calc_(data.second.data.data(), c.data(), dim);
              if (dis < min_dis) {
                  min_dis = dis;
                  min_index = static_cast<int>(i);
              }
          }
          ivf_clusters_[min_index]->add(data.second.data.data(), data.first, dim);
      }

  }


  template<typename vec_t>
  Status IvfIndex<vec_t>::search(const vec_t *query_vec, size_t k, std::vector<idx_t> &result_ids,
                                 std::vector<float> &result_distances) const {
      using wrap = std::pair<float, size_t>;

      bounded_priority_queue<wrap, std::greater<>> queue(header_.probes_);
      auto size = ivf_clusters_.size();
      auto dim = header_.dim_;
      auto dis_type = header_.distance_type_;

      for (size_t i = 0; i < size; ++i) {
          auto &cluster = ivf_clusters_[i];
          auto dis = calc_(query_vec, cluster->centroid().data(), dim);
          queue.push({dis, i});
      }

      bounded_priority_queue<predict_result, std::greater<>> result_queue(k);

      for (size_t i = 0; i < header_.probes_; ++i) {
          auto &cluster = ivf_clusters_[queue.top().second];

          result_queue.merge(cluster->predict(k, query_vec, dim, dis_type));
      }

      result_ids.clear();
      result_distances.clear();

      for (const auto &r: result_queue.dump()) {
          result_ids.push_back(r.id);
          result_distances.push_back(r.dis);
      }

      return Status::OK();
  }

  template<typename vec_t>
  size_t IvfIndex<vec_t>::dimension() const {
      return header_.dim_;
  }

  template<typename vec_t>
  size_t IvfIndex<vec_t>::size() const {
      return datas_.size();
  }

  template<typename vec_t>
  IvfIndex<vec_t>::IvfIndex(int lists, int probes, int dim, DistanceType type, ClusterType c_type)
          : header_{lists, probes, dim, type, c_type}, calc_{type}, kmeans_(lists, dim) {
  }


  template<typename vec_t>
  Status IvfIndex<vec_t>::add(idx_t id, const vec_t *vec_ptr) {
      auto dt = data_type<vec_t>(vec_ptr, vec_ptr + header_.dim_, id);
      kmeans_.add(dt.data.begin());
      datas_.emplace(id, data_type<vec_t>(vec_ptr, vec_ptr + header_.dim_, id));
      return Status::OK();
  }


}
