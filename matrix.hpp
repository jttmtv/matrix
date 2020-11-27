/* PAY ATTENTION:
Function "matrix_multiplication3/6": The rows and columns of the matrix to be calculated must be integer multiples of 64
Function "matrix_multiplication4/5": The rows and columns of the matrix to be calculated must be integer multiples of 8
*/

#ifndef _MATRIX_
#define _MATRIX_
#include <cstddef>

struct matrix
{
    size_t row;
    size_t col;
    float **val;
};

bool check(const matrix *, const matrix *);
void matrix_set(matrix *);
void matrix_clear(matrix *);
void matrix_transpose(const matrix *, matrix *);
void memory_access(matrix *, size_t, size_t);
void memory_free(matrix *);

void matrix_multiplication1(const matrix *, const matrix *, matrix *);
void matrix_multiplication2(const matrix *, const matrix *, matrix *);
void matrix_multiplication3(const matrix *, const matrix *, matrix *);
void matrix_multiplication4(const matrix *, const matrix *, matrix *);
void matrix_multiplication5(const matrix *, const matrix *, matrix *);
void matrix_multiplication6(const matrix *, const matrix *, matrix *);
#endif
