#include "matrix.hpp"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <ctime>

bool check(const matrix* A, const matrix* B)
{
    if (!A->row || !A->col || !B->row || !B->col)
        return false;
    else if (A->col != B->row)
        return false;
    else
        return true;
}

void matrix_set(matrix* X)
{
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < X->row; ++i)
        for (size_t j = 0; j < X->col; ++j)
            X->val[i][j] = rand() / float(RAND_MAX);
}

void memory_access(matrix* X, size_t row, size_t col)
{
    X->row = row;
    X->col = col;
    if (row > 0 && col > 0)
    {
        X->val = new float* [row];
        for (size_t i = 0; i < row; ++i)
            X->val[i] = new float[col];
    }
}

void memory_free(matrix* X)
{
    if (X->val)
    {
        for (size_t i = 0; i < X->row; ++i)
            delete[] X->val[i];
        delete[] X->val;
    }
}

void matrix_clear(matrix* X)
{
    for (size_t i = 0; i < X->row; ++i)
        memset(X->val[i], 0, sizeof(float) * X->col);
}

void matrix_transpose(const matrix* X, matrix* XT)
{
    for (size_t i = 0; i < X->row; ++i)
        for (size_t j = 0; j < X->col; ++j)
            XT->val[j][i] = X->val[i][j];
}

inline void block(const matrix *A, const matrix *B, matrix *C, size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < 64; ++i)
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < 64; ++k)
                C->val[i + M][k + N] += A->val[i + M][j + K] * B->val[j + K][k + N];
}

void matrix_multiplication1(const matrix *A, const matrix *B, matrix *C)
{
    for (size_t i = 0; i < C->row; ++i)
        for (size_t j = 0; j < C->col; ++j)
            for (size_t k = 0; k < A->col; ++k)
                C->val[i][j] += A->val[i][k] * B->val[k][j];
}

void matrix_multiplication2(const matrix *A, const matrix *B, matrix *C)
{
    for (size_t i = 0; i < C->row; ++i)
        for (size_t j = 0; j < B->row; ++j)
            for (size_t k = 0; k < C->col; ++k)
                C->val[i][k] += A->val[i][j] * B->val[j][k];
}

void matrix_multiplication3(const matrix *A, const matrix *B, matrix *C)
{
    for (size_t m = 0; m < C->row; m += 64)
        for (size_t n = 0; n < C->col; n += 64)
            for (size_t k = 0; k < A->col; k += 64)
                block(A, B, C, m, n, k);
}

void matrix_multiplication4(const matrix* A, const matrix* B, matrix* C)
{
    size_t m, n, k;
    for (m = 0; m < A->row; ++m)
    {
        for (k = 0; k < A->col; ++k)
        {
            __m256 v8_a = _mm256_set1_ps(A->val[m][k]);
            for (n = 0; n < B->col - 7; n += 8)
            {
                __m256 v8_b = _mm256_loadu_ps(B->val[k] + n);
                __m256 v8_c = _mm256_loadu_ps(C->val[m] + n);
                _mm256_storeu_ps(C->val[m] + n, _mm256_add_ps(v8_c, _mm256_mul_ps(v8_a, v8_b)));
            }
            for (; n < B->col; ++n)
                C->val[m][n] += A->val[m][k] * B->val[k][n];
        }
    }
}

void matrix_multiplication5(const matrix* A, const matrix* B, matrix* C)
{
    size_t m = 0, n = 0, k = 0;
    size_t M = A->row, N = B->row, K = A->col;
    __m256 v8_1_ps = _mm256_set1_ps(1.0f);
    __m256 v8_sum_tmp_ps, v8_sumv_tmp_ps;
    for (m = 0; m < M; ++m)
    {
        for (n = 0; n < N - 7; n += 8)
        {
            float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
            sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = 0.0f;
            __m256 v8_sum0 = _mm256_setzero_ps();
            __m256 v8_sum1 = _mm256_setzero_ps();
            __m256 v8_sum2 = _mm256_setzero_ps();
            __m256 v8_sum3 = _mm256_setzero_ps();
            __m256 v8_sum4 = _mm256_setzero_ps();
            __m256 v8_sum5 = _mm256_setzero_ps();
            __m256 v8_sum6 = _mm256_setzero_ps();
            __m256 v8_sum7 = _mm256_setzero_ps();

            for (k = 0; k < K - 7; k += 8)
            {
                __m256 a = _mm256_loadu_ps(&A->val[m][k]);

                __m256 b0 = _mm256_loadu_ps(&(B->val[n][k]));
                __m256 b1 = _mm256_loadu_ps(&(B->val[n + 1][k]));
                __m256 b2 = _mm256_loadu_ps(&(B->val[n + 2][k]));
                __m256 b3 = _mm256_loadu_ps(&(B->val[n + 3][k]));
                __m256 b4 = _mm256_loadu_ps(&(B->val[n + 4][k]));
                __m256 b5 = _mm256_loadu_ps(&(B->val[n + 5][k]));
                __m256 b6 = _mm256_loadu_ps(&(B->val[n + 6][k]));
                __m256 b7 = _mm256_loadu_ps(&(B->val[n + 7][k]));

                v8_sum0 = _mm256_add_ps(v8_sum0, _mm256_mul_ps(a, b0));
                v8_sum1 = _mm256_add_ps(v8_sum1, _mm256_mul_ps(a, b1));
                v8_sum2 = _mm256_add_ps(v8_sum2, _mm256_mul_ps(a, b2));
                v8_sum3 = _mm256_add_ps(v8_sum3, _mm256_mul_ps(a, b3));
                v8_sum4 = _mm256_add_ps(v8_sum4, _mm256_mul_ps(a, b4));
                v8_sum5 = _mm256_add_ps(v8_sum5, _mm256_mul_ps(a, b5));
                v8_sum6 = _mm256_add_ps(v8_sum6, _mm256_mul_ps(a, b6));
                v8_sum7 = _mm256_add_ps(v8_sum7, _mm256_mul_ps(a, b7));
            }
            for (; k < K; ++k)
            {
                sum0 += A->val[m][k] * B->val[n][k];
                sum1 += A->val[m][k] * B->val[n + 1][k];
                sum2 += A->val[m][k] * B->val[n + 2][k];
                sum3 += A->val[m][k] * B->val[n + 3][k];
                sum4 += A->val[m][k] * B->val[n + 4][k];
                sum5 += A->val[m][k] * B->val[n + 5][k];
                sum6 += A->val[m][k] * B->val[n + 6][k];
                sum7 += A->val[m][k] * B->val[n + 7][k];
            }
            v8_sum_tmp_ps = _mm256_setr_ps(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum0, v8_1_ps, 0xF1);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum1, v8_1_ps, 0xF2);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum2, v8_1_ps, 0xF4);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum3, v8_1_ps, 0xF8);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum4, v8_1_ps, 0xF16);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum5, v8_1_ps, 0xF32);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum6, v8_1_ps, 0xF64);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            v8_sumv_tmp_ps = _mm256_dp_ps(v8_sum7, v8_1_ps, 0xF128);
            v8_sum_tmp_ps = _mm256_add_ps(v8_sum_tmp_ps, v8_sumv_tmp_ps);

            _mm256_storeu_ps(&(C->val[m][n]), v8_sum_tmp_ps);
        }
        for (; n < N; ++n)
        {
            float sum0;
            __m256 v8_sum0 = _mm256_setzero_ps();
            sum0 = 0.0f;
            for (k = 0; k < K - 7; k += 8)
            {
                __m256 a = _mm256_loadu_ps(&(A->val[m][k]));
                __m256 b0 = _mm256_loadu_ps(&(B->val[n][k]));
                v8_sum0 = _mm256_add_ps(v8_sum0, _mm256_mul_ps(a, b0));
            }
            for (; k < K; k++)
                sum0 += A->val[m][k] * B->val[n][k];
            C->val[m][n] = sum0 + v8_sum0.m256_f32[0] + v8_sum0.m256_f32[1] + v8_sum0.m256_f32[2] + v8_sum0.m256_f32[3] + v8_sum0.m256_f32[4] + v8_sum0.m256_f32[5] + v8_sum0.m256_f32[6] + v8_sum0.m256_f32[7];
        }
    }
}
