#include "validator.h"

bool MultiplicationValidator( const Matrix_t& left, const Matrix_t& right )
{
    if( left.cols != right.rows )
    {
        return false;
    }

    return true;
}

bool AdditionValidator( const Matrix_t& left, const Matrix_t& right )
{
    if( left.rows == right.rows && left.cols == right.cols )
    {
        return false;
    }

    return true;
}
