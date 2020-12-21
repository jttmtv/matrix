#include <omp.h>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cblas.h>
#include <iostream>
#include <exception>
#include <immintrin.h>
using namespace std;
using namespace chrono;
typedef long long llong;
typedef unsigned long long ullong;
typedef unsigned int uint;
typedef unsigned char uchar;
#define NUM_THREADS 12

inline void core_cb(const float* rowA, const float* rowB, float* matC, const size_t K, const size_t N)
{
	__m256 reg_a = _mm256_setzero_ps();
	__m256 reg_b = _mm256_setzero_ps();
	__m256 reg_c00 = _mm256_setzero_ps();
	__m256 reg_c10 = _mm256_setzero_ps();
	__m256 reg_c20 = _mm256_setzero_ps();
	__m256 reg_c30 = _mm256_setzero_ps();
	__m256 reg_c40 = _mm256_setzero_ps();
	__m256 reg_c50 = _mm256_setzero_ps();
	__m256 reg_c60 = _mm256_setzero_ps();
	__m256 reg_c70 = _mm256_setzero_ps();
	const float* ptr_a = rowA;
	const float* ptr_b = rowB;
	for (size_t k = 0; k < K; ++k)
	{
		reg_b = _mm256_loadu_ps(ptr_b);

		reg_a = _mm256_set1_ps(ptr_a[0]);
		reg_c00 = _mm256_add_ps(reg_c00, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[1]);
		reg_c10 = _mm256_add_ps(reg_c10, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[2]);
		reg_c20 = _mm256_add_ps(reg_c20, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[3]);
		reg_c30 = _mm256_add_ps(reg_c30, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[4]);
		reg_c40 = _mm256_add_ps(reg_c40, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[5]);
		reg_c50 = _mm256_add_ps(reg_c50, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[6]);
		reg_c60 = _mm256_add_ps(reg_c60, _mm256_mul_ps(reg_a, reg_b));

		reg_a = _mm256_set1_ps(ptr_a[7]);
		reg_c70 = _mm256_add_ps(reg_c70, _mm256_mul_ps(reg_a, reg_b));

		ptr_a += 8;
		ptr_b += 8;
	}
	_mm256_storeu_ps(matC + 0LL * N, reg_c00);
	_mm256_storeu_ps(matC + 1LL * N, reg_c10);
	_mm256_storeu_ps(matC + 2LL * N, reg_c20);
	_mm256_storeu_ps(matC + 3LL * N, reg_c30);
	_mm256_storeu_ps(matC + 4LL * N, reg_c40);
	_mm256_storeu_ps(matC + 5LL * N, reg_c50);
	_mm256_storeu_ps(matC + 6LL * N, reg_c60);
	_mm256_storeu_ps(matC + 7LL * N, reg_c70);
}

void core_mu(const float* matA, const float* matB, float* matC, const size_t M, const size_t K, const size_t N)
{
	try
	{
		size_t M8 = M - M % 8;
		float* packA = new float[M8 * K];
		float* dst_unit = packA;
		float* src_unit;
		__m256 src_data;
		for (size_t i = 0; i < M8; i += 8) {
			for (size_t k = 0; k < K; ++k) {
				src_unit = const_cast<float*>(&matA[i * K + k]);
				src_data = _mm256_set_ps(src_unit[7 * K], src_unit[6 * K], src_unit[5 * K], src_unit[4 * K], src_unit[3 * K], src_unit[2 * K], src_unit[1 * K], src_unit[0 * K]);
				_mm256_storeu_ps(dst_unit, src_data);
				dst_unit += 8;
			}
		}
		size_t N8 = N - N % 8;
		float* packB = new float[K * N8];
		dst_unit = packB;
		for (size_t j = 0; j < N8; j += 8) {
			for (size_t k = 0; k < K; ++k) {
				_mm256_storeu_ps(dst_unit, _mm256_loadu_ps(&matB[k * N + j]));
				dst_unit += 8;
			}
		}
#ifdef _OPENMP
		omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
#endif
		for (size_t j = 0; j < N8; j += 8)
			for (size_t i = 0; i < M8; i += 8)
				core_cb(&packA[i * K], &packB[j * K], &matC[i * N + j], K, N);
		delete[] packA;
		delete[] packB;
	}
	catch (bad_alloc& ME) {
		cerr << ME.what() << endl;
	}
}

void core_ma(const float* matA, const float* matB, float* matC, const size_t M, const size_t K, const size_t N)
{
	const size_t M8 = M - M % 8, N8 = N - N % 8;
	core_mu(matA, matB, matC, M, K, N);
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
#endif
	for (size_t m = M8; m < M; ++m)
		for (size_t k = 0; k < K; ++k)
			for (size_t n = 0; n < N; ++n)
				matC[m * N + n] += matA[m * K + k] * matB[k * N + n];
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
#endif
	for (size_t m = 0; m < M8; ++m)
		for (size_t k = 0; k < K; ++k)
			for (size_t n = N8; n < N; ++n)
				matC[m * N + n] += matA[m * K + k] * matB[k * N + n];
}

void core_sm(const float* matA, const float* matB, float* matC, const size_t M, const size_t K, const size_t N)
{
#ifdef _OPENMP
	omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
#endif
	for (size_t m = 0; m < M; ++m)
		for (size_t k = 0; k < K; ++k)
			for (size_t n = 0; n < N; ++n)
				matC[m * N + n] += matA[m * K + k] * matB[k * N + n];
}

bool fast_sgemm(const float* A, const float* B, float* C, const size_t M, const size_t K, const size_t N)
{
	if (!(M && K && N))
		return false;
	memset(C, 0, sizeof(float) * M * N);
	if (!(M % 8 || N % 8))
		core_mu(A, B, C, M, K, N);
	else if (M < 8 && N < 8)
		core_sm(A, B, C, M, K, N);
	else
		core_ma(A, B, C, M, K, N);
	return true;
}

int main()
{
	const size_t M = 10000;
	const size_t K = 3;
	const size_t N = 10000;
	float* A = new float[M * K];
	float* B = new float[K * N];
	float* C = new float[M * N];
	float* CC = new float[M * N];
	memset(C, 0, sizeof(float) * M * N);
	memset(CC, 0, sizeof(float) * M * N);
	srand((unsigned)time(NULL));
	for (size_t i = 0; i < M * K; i++)
		A[i] = rand() / float(RAND_MAX);
	for (size_t i = 0; i < K * N; i++)
		B[i] = rand() / float(RAND_MAX);
	auto start = system_clock::now();
	fast_sgemm(A, B, C, M, K, N);
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, CC, N);
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << double(duration.count()) * microseconds::period::num / microseconds::period::den << "s" << endl;
	core_sm(A, B, CC, M, K, N);
	for (size_t i = 0; i < M * N; i++)
	{
		if (abs(C[i] - CC[i]) > 1e-6f)
			cout << i << endl;
	}
	cout << "OK" << endl;
	//start = system_clock::now();
	//fast_sgemm(A, B, C, M, K, N);
	//end = system_clock::now();
	//duration = duration_cast<microseconds>(end - start);
	//cout << double(duration.count()) * microseconds::period::num / microseconds::period::den << "s" << endl;
	delete[] A;
	delete[] B;
	delete[] C;
	return 0;
}
