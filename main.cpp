#include "matrix.hpp"
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

    // for (int i = 0; i < 8; i++)
    //     for (int j = 0; j < 8; j++)
    //         B.val[i][j] = j;
    // for (int i = 0; i < 8; i++)
    //     for (int j = 0; j < 8; j++)
    //         A.val[i][j] = j;
    if (check(&A, &B))
    {
        matrix_clear(&C);
        matrix_transpose(&B, &BT);
        auto start = std::chrono::steady_clock::now();
        matrix_multiplication5(&A, &BT, &C);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "total times have been used:" << duration << "ms" << std::endl;
    }
    else
        std::cerr << "ILLEGAL DATA" << std::endl;

    // for (int i = 0; i < 8; i++)
    // {
    //     for (int j = 0; j < 8; j++)
    //         std::cout << C.val[i][j] << " ";
    //     std::cout << std::endl;
    // }

    memory_free(&A);
    memory_free(&B);
    memory_free(&C);
    memory_free(&BT);
    return 0;
}