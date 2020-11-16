#include "matrix.h"
#include <chrono>
#include <iostream>
#define ROW_SIZE 1000
#define COL_SIZE 1000
#define SHARE_SIZE 1000

int main()
{
    matrix A, B, BT, C;

    memory_access(&A, ROW_SIZE, SHARE_SIZE);
    memory_access(&B, SHARE_SIZE, COL_SIZE);
    memory_access(&C, ROW_SIZE, COL_SIZE);
    memory_access(&BT, COL_SIZE, SHARE_SIZE);

    matrix_set(&A);
    matrix_set(&B);

    std::cout<< "1: Brute force\n"
        << "2: Change the order of the loop\n"
        << "3: Blocked Matrix\n"
        << "4: Preliminary AVX instruction set\n"
        << "5: UNROOL + AVX\n"
        << "6: OpenMp + AVX\n"
        << std::endl;
        if (check(&A, &B))
        {
            matrix_clear(&C);
            auto start = std::chrono::steady_clock::now();
            matrix_multiplication1(&A, &B, &C);
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution1 total times have been used:" << duration << "ms" << std::endl;

            matrix_clear(&C);
            start = std::chrono::steady_clock::now();
            matrix_multiplication2(&A, &B, &C);
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution2 total times have been used:" << duration << "ms" << std::endl;

            matrix_clear(&C);
            start = std::chrono::steady_clock::now();
            matrix_multiplication3(&A, &B, &C);
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution3 total times have been used:" << duration << "ms" << std::endl;

            matrix_clear(&C);
            start = std::chrono::steady_clock::now();
            matrix_multiplication4(&A, &B, &C);
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution4 total times have been used:" << duration << "ms" << std::endl;

            matrix_clear(&C);
            matrix_transpose(&B, &BT);
            start = std::chrono::steady_clock::now();
            matrix_multiplication5(&A, &BT, &C);
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution5 total times have been used:" << duration << "ms" << std::endl;

            matrix_clear(&C);
            matrix_transpose(&B, &BT);
            start = std::chrono::steady_clock::now();
            matrix_multiplication5(&A, &BT, &C);
            end = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Solution6 total times have been used:" << duration << "ms" << std::endl;
        }
        else
            std::cerr << "ILLEGAL DATA" << std::endl;
    memory_free(&A);
    memory_free(&B);
    memory_free(&C);
    memory_free(&BT);
    return 0;
}
