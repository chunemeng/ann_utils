#pragma once

#include <cmath>

namespace alp {
  enum DistanceType {
      UNKNOWN = 0,
      L2 = 1,
      IP = 2,
      COSINE = 3,
  };

  template<typename vec_t>
  static constexpr inline float l2_distance(const vec_t *a, const vec_t *b, int size) {
      float sum = 0;
      for (int i = 0; i < size; i++) {
          float diff = a[i] - b[i];
          sum += diff * diff;
      }
      return sum;

//          __m256 sum_vec = _mm256_setzero_ps();
//          int i;
//
//          // 处理 8 个 float 数据
//          for (i = 0; i <= size - 8; i += 8) {
//              __m256 a_vec = _mm256_loadu_ps(&a[i]);
//              __m256 b_vec = _mm256_loadu_ps(&b[i]);
//              __m256 diff_vec = _mm256_sub_ps(a_vec, b_vec);
//              sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff_vec, diff_vec));
//          }
//
//          float sum[8];
//          _mm256_storeu_ps(sum, sum_vec);
//
//          float total_sum = 0.0f;
//          for (int j = 0; j < 8; j++) {
//              total_sum += sum[j];
//          }
//
//          for (; i < size; i++) {
//              float diff = a[i] - b[i];
//              total_sum += diff * diff;
//          }
//
//          return total_sum;
  }

  template<typename vec_t>
  static constexpr inline float ip_distance(const vec_t *a, const vec_t *b, int size) {
      float sum = 0;
      for (int i = 0; i < size; i++) {
          sum += a[i] * b[i];
      }
      return sum;
  }


  template<typename vec_t>
  static constexpr inline float cosine_distance(const vec_t *a, const vec_t *b, int size) {
      float dot_product = 0;
      float norm_a = 0;
      float norm_b = 0;
      for (int i = 0; i < size; i++) {
          dot_product += a[i] * b[i];
          norm_a += a[i] * a[i];
          norm_b += b[i] * b[i];
      }
      return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
  }

  static inline constexpr float EPSILON = 1e-6f;

  template<typename vec_t>
  class DistanceCalc {
  public:
      DistanceCalc() = default;

      explicit DistanceCalc(DistanceType type) {
          switch (type) {
              case DistanceType::L2:
                  calc = l2_distance<vec_t>;
                  break;
              case DistanceType::IP:
                  calc = ip_distance<vec_t>;
                  break;
              case DistanceType::COSINE:
                  calc = cosine_distance<vec_t>;
                  break;
              default:
                  calc = l2_distance<vec_t>;
                  break;
          }
      }

      void init(DistanceType type) {
          switch (type) {
              case DistanceType::L2:
                  calc = l2_distance<vec_t>;
                  break;
              case DistanceType::IP:
                  calc = ip_distance<vec_t>;
                  break;
              case DistanceType::COSINE:
                  calc = cosine_distance<vec_t>;
                  break;
              default:
                  calc = l2_distance<vec_t>;
                  break;
          }
      }

      float operator()(const vec_t *a, const vec_t *b, int size) const { return calc(a, b, size); }

      static int compare(const vec_t &a, const vec_t &b) {
          float cmp = a - b;
          if (cmp > EPSILON) {
              return 1;
          }
          if (cmp < -EPSILON) {
              return -1;
          }
          return 0;
      }

      static int compare(const vec_t *a, const vec_t *b, int size) {
          for (int i = 0; i < size; i++) {
              int result = compare(a[i], b[i]);
              if (result != 0) {
                  return result;
              }
          }
          return 0;
      }

  private:
      float (*calc)(const vec_t *a, const vec_t *b, int size) = nullptr;
  };
}

