#pragma once

#include <queue>

namespace alp {

  template<typename T, typename Compare = std::less<T>>
  class bounded_priority_queue {
  public:
      using priority_queue_type = std::priority_queue<T, std::vector<T>, Compare>;

      explicit bounded_priority_queue(size_t max_size) : max_size_(max_size) {
          queue_.reserve(max_size);
      }

      bounded_priority_queue() = delete;

      void push(const T &value) {
          if (queue_.size() < max_size_) {
              queue_.push(value);
          } else if (Compare()(value, queue_.top())) {
              queue_.pop();
              queue_.push(value);
          }
      }

      std::vector<T> dump() {
          std::vector<T> result;
          result.reserve(queue_.size());
          while (!queue_.empty()) {
              result.push_back(queue_.top());
              queue_.pop();
          }
          return result;
      }

      void merge(bounded_priority_queue &other) {
          while (!other.empty()) {
              push(other.top());
              other.pop();
          }
      }

      void pop() {
          if (!queue_.empty()) {
              queue_.pop();
          }
      }

      const T &top() const {
          return queue_.top();
      }


      bool empty() const {
          return queue_.empty();
      }

      void clear() {
          while (!queue_.empty()) {
              queue_.pop();
          }
      }

      size_t size() const {
          return queue_.size();
      }

      priority_queue_type &queue() {
          return queue_;
      }

  private:
      priority_queue_type queue_;

      size_t max_size_;


  };

} // namespace alp

