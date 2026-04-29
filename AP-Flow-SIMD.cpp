// Tatiana Melnichenko
// COSC 594, Lab 4: SIMD AP Flow
// April 29, 2026

#include <emmintrin.h>
#include "AP-Flow.h"

void APFlow::CalcFlow()
{
  for (int i = 0; i < N * N; ++i) {
    Flow[i] = Adj[i];
  }

  for (int v = 0; v < N; ++v) {
    for (int i = 0; i < N; ++i) {
      // alli: 16 instances of Flow[i][v]
      __m128i alli = _mm_set1_epi8(Flow[i * N + v]);
      for (int j = 0; j < N; j += 16) {
        // Load Flow[v][j] though Flow[v][j+15] into vv
        __m128i vv = _mm_loadu_si128((__m128i const*)&Flow[v * N + j]);
        // fv: flow from i to each of j through j+15 through v
        __m128i fv = _mm_min_epu8(alli, vv);
        // Load Flow[i][j] though Flow[i][j+15] into iv
        __m128i iv = _mm_loadu_si128((__m128i const*)&Flow[i * N + j]);
        // rv: best flow values between fv and iv
        __m128i rv = _mm_max_epu8(fv, iv);
        // Store result
        _mm_storeu_si128((__m128i*)&Flow[i * N + j], rv);
      }
    }
  }
}
