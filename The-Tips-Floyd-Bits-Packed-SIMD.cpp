// Tatiana Melnichenko
// COSC 594, Lab 4: SIMD Tips
// April 29, 2026

#include <emmintrin.h>
#include "The-Tips.h"

double TheTips::solve(vector <string> clues, vector <int> probability, int print)
{
  size_t num_eggs = probability.size();
  // Make row size evenly divisible by 16 for SIMD
  size_t num_eggs_padded = (num_eggs + 15) % ~15;

  // Build the adjacency matrix
  vector<vector<char>> C(num_eggs);
  for (int i = 0; i < num_eggs; ++i) {
    C[i].resize(num_eggs_padded, 0);
    for (int j = 0; j < num_eggs; ++j) {
      C[i][j] = (i == j || clues[i][j] == 'Y') ? 1 : 0;
    }
  }

  if (print) {
    printf("The Adjacency Matrix:\n\n    ");
    for (int i = 0; i < num_eggs; ++i) {
      printf("%X", i % 16);
    }
    printf("\n   ");
    for (int i = 0; i <= num_eggs; ++i) cout << "-";
    printf("\n");
    for (int i = 0; i < num_eggs; ++i) {
      printf("%-2X| ", i % 16);
      for (int j = 0; j < num_eggs; ++j) {
        printf("%d", C[i][j]);
      }
      printf("\n");
    }
  }

  // Use Floyd-Warshall to build the connectivity matrix
  for (int v = 0; v < num_eggs; ++v) {
    for (int i = 0; i < num_eggs; ++i) {
      if (C[i][v]) {
        // 128 bits = 16 bytes
        // Rows are padded with zeroes, so loads and stores don't fail when num_eggs % 16 != 0
        for (int j = 0; j < num_eggs; j += 16) {
          __m128i vA = _mm_loadu_si128((__m128i const*)&C[i][j]);
          __m128i vB = _mm_loadu_si128((__m128i const*)&C[v][j]);
          __m128i vRes = _mm_or_si128(vA, vB);
          _mm_storeu_si128((__m128i*)&C[i][j], vRes);
        }
      }
    }
  }

  if (print) {
    printf("\nProbabilities:\n\n");
    for (int i = 0; i < num_eggs; ++i) printf("%4d", probability[i]);
    printf("\n\n");

    printf("The Floyd-Warshall Matrix:\n\n    ");
    for (int i = 0; i < num_eggs; ++i) {
      printf("%X", i % 16);
    }
    printf("\n   ");
    for (int i = 0; i <= num_eggs; ++i) cout << "-";
    printf("\n");
    for (int i = 0; i < num_eggs; ++i) {
      printf("%-2X| ", i % 16);
      for (int j = 0; j < num_eggs; ++j) {
        printf("%d", C[i][j]);
      }
      printf("\n");
    }
  }

  // Calculate the probabilities for each egg
  vector<double> p(num_eggs, 0.0);
  for (int i = 0; i < num_eggs; ++i) {
    double tmp = probability[i] / 100.0;
    for (int j = 0; j < num_eggs; ++j) {
      if (C[i][j]) {
        p[j] += ((1 - p[j]) * tmp);
      }
    }
  }

  if (print) {
    printf("\n\nThe Expected Values\n\n");
    for (int i = 0; i < num_eggs; ++i) {
      printf("I: %X    Prob: %.5f\n", i, p[i]);
    }
    printf("\n");
  }

  // Calculate the final return value
  double sum = 0.0;
  for (int i = 0; i < num_eggs; ++i) {
    sum += p[i];
  }

  return sum;
}
