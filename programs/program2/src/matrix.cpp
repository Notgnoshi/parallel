#include "matrix.h"
#include <cstddef>
#include <cstdint>
#include <fstream>

Matrix_t Deserialize( std::string filename )
{
    std::ifstream file( filename, std::ios::binary );

    size_t rows = 0;
    size_t cols = 0;
    Matrix_t matrix( rows, cols );

    return matrix;
}

void Serialize( Matrix_t matrix, std::string filename )
{
    std::ofstream file( filename, std::ofstream::binary | std::ofstream::trunc );

    file.write( reinterpret_cast<const char*>( &matrix.rows ), sizeof( size_t ) );
    file.write( reinterpret_cast<const char*>( &matrix.cols ), sizeof( size_t ) );

    for( auto const& row : matrix.matrix )
    {
        file.write( reinterpret_cast<const char*>( row.data() ), sizeof( double ) * row.size() );
    }

    file.close();
}
