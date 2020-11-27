#include "matrix.hpp"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <iostream>
#include <omp.h>

bool check(const matrix *A, const matrix *B)
{
    if (!A->row || !A->col || !B->row || !B->col)
        return false;
    else if (A->col != B->row)
        return false;
    else
        return true;
}

void matrix_set(matrix *X)
{
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < X->row; ++i)
        for (size_t j = 0; j < X->col; ++j)
            X->val[i][j] = rand() / float(RAND_MAX);
}

void memory_access(matrix *X, size_t row, size_t col)
{
    X->row = row;
    X->col = col;
    if (row > 0 && col > 0)
    {
        X->val = new float *[row];
        for (size_t i = 0; i < row; ++i)
            X->val[i] = new float[col];
    }
}

void memory_free(matrix *X)
{
    if (X->val)
    {
        for (size_t i = 0; i < X->row; ++i)
            delete[] X->val[i];
        delete[] X->val;
    }
}

void matrix_clear(matrix *X)
{
    for (size_t i = 0; i < X->row; ++i)
        memset(X->val[i], 0, sizeof(float) * X->col);
}

void matrix_transpose(const matrix *X, matrix *XT)
{
    for (size_t i = 0; i < X->row; ++i)
        for (size_t j = 0; j < X->col; ++j)
            XT->val[j][i] = X->val[i][j];
}

inline void block(const matrix *A, const matrix *B, matrix *C, size_t M, size_t N, size_t K)
{
    for (size_t m = 0; m < 64; ++m)
        for (size_t k = 0; k < 64; ++k)
            for (size_t n = 0; n < 64; ++n)
                C->val[m + M][n + N] += A->val[m + M][k + K] * B->val[k + K][n + N];
}

inline void block_avx(const matrix *A, const matrix *B, matrix *C, size_t M, size_t N, size_t K)
{
    float dp[8][8];
    __m256 a[8], b[8], c[8];
    for (size_t m = 0; m < 64; ++m)
    {
        a[0] = _mm256_loadu_ps(&A->val[M + m][K]);
        a[1] = _mm256_loadu_ps(&A->val[M + m][K + 8]);
        a[2] = _mm256_loadu_ps(&A->val[M + m][K + 16]);
        a[3] = _mm256_loadu_ps(&A->val[M + m][K + 24]);
        a[4] = _mm256_loadu_ps(&A->val[M + m][K + 32]);
        a[5] = _mm256_loadu_ps(&A->val[M + m][K + 40]);
        a[6] = _mm256_loadu_ps(&A->val[M + m][K + 48]);
        a[7] = _mm256_loadu_ps(&A->val[M + m][K + 56]);
        for (size_t n = 0; n < 64; ++n)
        {
            b[0] = _mm256_loadu_ps(&B->val[N + n][K]);
            b[1] = _mm256_loadu_ps(&B->val[N + n][K + 8]);
            b[2] = _mm256_loadu_ps(&B->val[N + n][K + 16]);
            b[3] = _mm256_loadu_ps(&B->val[N + n][K + 24]);
            b[4] = _mm256_loadu_ps(&B->val[N + n][K + 32]);
            b[5] = _mm256_loadu_ps(&B->val[N + n][K + 40]);
            b[6] = _mm256_loadu_ps(&B->val[N + n][K + 48]);
            b[7] = _mm256_loadu_ps(&B->val[N + n][K + 56]);

            c[0] = _mm256_dp_ps(a[0], b[0], 0xF1);
            c[1] = _mm256_dp_ps(a[1], b[1], 0xF1);
            c[2] = _mm256_dp_ps(a[2], b[2], 0xF1);
            c[3] = _mm256_dp_ps(a[3], b[3], 0xF1);
            c[4] = _mm256_dp_ps(a[4], b[4], 0xF1);
            c[5] = _mm256_dp_ps(a[5], b[5], 0xF1);
            c[6] = _mm256_dp_ps(a[6], b[6], 0xF1);
            c[7] = _mm256_dp_ps(a[7], b[7], 0xF1);

            _mm256_storeu_ps(dp[0], c[0]);
            _mm256_storeu_ps(dp[1], c[1]);
            _mm256_storeu_ps(dp[2], c[2]);
            _mm256_storeu_ps(dp[3], c[3]);
            _mm256_storeu_ps(dp[4], c[4]);
            _mm256_storeu_ps(dp[5], c[5]);
            _mm256_storeu_ps(dp[6], c[6]);
            _mm256_storeu_ps(dp[7], c[7]);
            C->val[M + m][N + n] += dp[0][0] + dp[0][4] + dp[1][0] + dp[1][4] + dp[2][0] + dp[2][4] + dp[3][0] + dp[3][4] + dp[4][0] + dp[4][4] + dp[5][0] + dp[5][4] + dp[6][0] + dp[6][4] + dp[7][0] + dp[7][4];
        }
    }
}

void matrix_multiplication1(const matrix *A, const matrix *B, matrix *C)
{
    for (size_t m = 0; m < C->row; ++m)
        for (size_t n = 0; n < C->col; ++n)
            for (size_t k = 0; k < A->col; ++k)
                C->val[m][n] += A->val[m][k] * B->val[k][n];
}

void matrix_multiplication2(const matrix *A, const matrix *B, matrix *C)
{
    for (size_t m = 0; m < C->row; ++m)
        for (size_t k = 0; k < B->row; ++k)
            for (size_t n = 0; n < C->col; ++n)
                C->val[m][n] += A->val[m][k] * B->val[k][n];
}

void matrix_multiplication3(const matrix *A, const matrix *B, matrix *C)
{

    for (size_t m = 0; m < C->row; m += 64)
        for (size_t n = 0; n < C->col; n += 64)
            for (size_t k = 0; k < A->col; k += 64)
                block(A, B, C, m, n, k);
}

void matrix_multiplication4(const matrix *A, const matrix *B, matrix *C)
{
    size_t m, n, k;
    for (m = 0; m < A->row; ++m)
    {
        for (k = 0; k < A->col; ++k)
        {
            __m256 v8_a = _mm256_set1_ps(A->val[m][k]);
            for (n = 0; n < B->col - 7; n += 8)
            {
                __m256 v8_b = _mm256_loadu_ps(&B->val[k][n]);
                __m256 v8_c = _mm256_loadu_ps(&C->val[m][n]);
                _mm256_storeu_ps(C->val[m] + n, _mm256_add_ps(v8_c, _mm256_mul_ps(v8_a, v8_b)));
            }
            for (; n < B->col; ++n)
                C->val[m][n] += A->val[m][k] * B->val[k][n];
        }
    }
}

void matrix_multiplication5(const matrix *A, const matrix *B, matrix *C)
{
    float sum[8];
    __m256 a, b[8], sumv[8];
    for (size_t m = 0; m < A->row; ++m)
    {
        for (size_t n = 0; n < B->col - 7; n += 8)
        {
            for (size_t k = 0; k < A->col - 7; k += 8)
            {
                a = _mm256_loadu_ps(&(A->val[m][k]));

                b[0] = _mm256_loadu_ps(&B->val[n][k]);
                b[1] = _mm256_loadu_ps(&B->val[n + 1][k]);
                b[2] = _mm256_loadu_ps(&B->val[n + 2][k]);
                b[3] = _mm256_loadu_ps(&B->val[n + 3][k]);
                b[4] = _mm256_loadu_ps(&B->val[n + 4][k]);
                b[5] = _mm256_loadu_ps(&B->val[n + 5][k]);
                b[6] = _mm256_loadu_ps(&B->val[n + 6][k]);
                b[7] = _mm256_loadu_ps(&B->val[n + 7][k]);

                sumv[0] = _mm256_dp_ps(a, b[0], 0xF1);
                sumv[1] = _mm256_dp_ps(a, b[1], 0xF1);
                sumv[2] = _mm256_dp_ps(a, b[2], 0xF1);
                sumv[3] = _mm256_dp_ps(a, b[3], 0xF1);
                sumv[4] = _mm256_dp_ps(a, b[4], 0xF1);
                sumv[5] = _mm256_dp_ps(a, b[5], 0xF1);
                sumv[6] = _mm256_dp_ps(a, b[6], 0xF1);
                sumv[7] = _mm256_dp_ps(a, b[7], 0xF1);

                _mm256_storeu_ps(sum, sumv[0]);
                C->val[m][n] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[1]);
                C->val[m][n + 1] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[2]);
                C->val[m][n + 2] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[3]);
                C->val[m][n + 3] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[4]);
                C->val[m][n + 4] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[5]);
                C->val[m][n + 5] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[6]);
                C->val[m][n + 6] += sum[0] + sum[4];

                _mm256_storeu_ps(sum, sumv[7]);
                C->val[m][n + 7] += sum[0] + sum[4];
            }
        }
    }
}

void matrix_multiplication6(const matrix *A, const matrix *B, matrix *C)
{
#pragma omp parallel for
    for (size_t m = 0; m < C->row; m += 64)
        for (size_t n = 0; n < C->col; n += 64)
            for (size_t k = 0; k < A->col; k += 64)
                block_avx(A, B, C, m, n, k);
}
