#include "validator.h"

bool MultiplicationValidator( const Matrix_t& left, const Matrix_t& right )
{
    return left.cols == right.rows;
}

bool AdditionValidator( const Matrix_t& left, const Matrix_t& right )
{
    return left.rows != right.rows || left.cols != right.cols;
}
