#include "matrix.h"
#include <cstddef>
#include <cstdint>
#include <fstream>

Matrix_t::Matrix_t( size_t rows, size_t cols ) :
    rows( rows ),
    cols( cols ),
    data( rows, std::vector<double>( cols, 0 ) )
{
}

Matrix_t::Matrix_t( const std::string& filename )
{
    this->rows = 0;
    this->cols = 0;

    std::ifstream file( filename, std::ios::binary );

    //! @todo Check the size of the file.

    file.read( reinterpret_cast<char*>( &this->rows ), sizeof( size_t ) );
    file.read( reinterpret_cast<char*>( &this->cols ), sizeof( size_t ) );

    this->data.resize( this->rows );

    for( auto& row : this->data )
    {
        row.resize( this->cols );
        file.read( reinterpret_cast<char*>( row.data() ), sizeof( double ) * row.size() );
    }
}

void Matrix_t::Serialize( const std::string& filename )
{
    std::ofstream file( filename, std::ofstream::binary | std::ofstream::trunc );

    file.write( reinterpret_cast<const char*>( &this->rows ), sizeof( size_t ) );
    file.write( reinterpret_cast<const char*>( &this->cols ), sizeof( size_t ) );

    for( auto const& row : this->data )
    {
        file.write( reinterpret_cast<const char*>( row.data() ), sizeof( double ) * row.size() );
    }

    file.close();
}

bool Matrix_t::operator==( const Matrix_t& other ) const
{
    return this->rows == other.rows && this->cols == other.cols && this->data == other.data;
}
