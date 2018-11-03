#include "matrix.h"
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>

Matrix_t::Matrix_t( size_t rows, size_t cols ) :
    rows( rows ),
    cols( cols ),
    elements( rows * cols )
{
    this->data = Allocate2D( rows, cols );

    if( this->data == nullptr )
    {
        this->rows = 0;
        this->cols = 0;
        this->elements = 0;
    }
}

Matrix_t::~Matrix_t()
{
    delete[] this->data;
}

Matrix_t::Matrix_t( const std::string& filename )
{
    std::ifstream file( filename, std::ios::binary );

    file.seekg( 0, std::ifstream::end );
    size_t size = file.tellg();
    file.seekg( 0, std::ifstream::beg );

    // Make sure there's enough of the file to read in the dimensions.
    if( size < 2 * sizeof( size_t ) )
    {
        return;
    }

    file.read( reinterpret_cast<char*>( &this->rows ), sizeof( size_t ) );
    file.read( reinterpret_cast<char*>( &this->cols ), sizeof( size_t ) );
    this->elements = this->rows * this->cols;

    // Make sure there's enough of the file to read in the dimensions and the data.
    if( size != 2 * sizeof( size_t ) + this->elements * sizeof( double ) )
    {
        this->rows = 0;
        this->cols = 0;
        this->elements = 0;
        return;
    }

    this->data = Allocate2D( this->rows, this->cols );
    // Make sure we can allocate enough memory.
    if( this->data == nullptr )
    {
        this->rows = 0;
        this->cols = 0;
        this->elements = 0;
        return;
    }

    file.read( reinterpret_cast<char*>( this->data ), sizeof( double ) * this->elements );
}

void Matrix_t::Serialize( const std::string& filename )
{
    std::ofstream file( filename, std::ofstream::binary | std::ofstream::trunc );

    file.write( reinterpret_cast<const char*>( &this->rows ), sizeof( size_t ) );
    file.write( reinterpret_cast<const char*>( &this->cols ), sizeof( size_t ) );
    file.write( reinterpret_cast<const char*>( this->data ), sizeof( double ) * this->elements );

    file.close();
}

void Matrix_t::Print() const
{
    std::cout << std::fixed << std::setprecision( 3 );
    for( size_t r = 0; r < this->rows; ++r )
    {
        for( size_t c = 0; c < this->cols; ++c )
        {
            std::cout << std::setw( 8 ) << this->operator()( r, c );
        }
        std::cout << std::endl;
    }
}

bool Matrix_t::operator==( const Matrix_t& other ) const
{
    // Dimensions need to match to be equal.
    if( this->rows != other.rows || this->cols != other.cols || this->elements != other.elements )
    {
        return false;
    }

    // Data needs to match to be equal too.
    for( size_t i = 0; i < this->rows * this->cols; ++i )
    {
        if( this->data[i] != other.data[i] )
        {
            return false;
        }
    }

    return true;
}

double& Matrix_t::operator()( size_t row, size_t col ) const
{
    return this->data[row * this->cols + col];
}

double& Matrix_t::operator()( size_t row, size_t col )
{
    return this->data[row * this->cols + col];
}

double* Matrix_t::Allocate2D( size_t rows, size_t cols )
{
    auto data = new double[rows * cols];

    if( data != nullptr )
    {
        for( size_t i = 0; i < rows * cols; ++i )
        {
            data[i] = 0.0;
        }
    }

    return data;
}
