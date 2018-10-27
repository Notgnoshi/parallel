#include "matrix.h"
#include <cstddef>
#include <cstdint>
#include <fstream>

std::shared_ptr<Matrix_t> Deserialize( const std::string& filename )
{
    std::ifstream file( filename, std::ios::binary );

    size_t rows = 0;
    size_t cols = 0;

    file.read( reinterpret_cast<char*>( &rows ), sizeof( size_t ) );
    file.read( reinterpret_cast<char*>( &cols ), sizeof( size_t ) );

    auto matrix = std::make_shared<Matrix_t>( rows, cols );

    for( auto& row : matrix->data )
    {
        file.read( reinterpret_cast<char*>( row.data() ), sizeof( double ) * row.size() );
    }

    return matrix;
}

void Serialize( const std::shared_ptr<Matrix_t>& matrix, const std::string& filename )
{
    std::ofstream file( filename, std::ofstream::binary | std::ofstream::trunc );

    file.write( reinterpret_cast<const char*>( &matrix->rows ), sizeof( size_t ) );
    file.write( reinterpret_cast<const char*>( &matrix->cols ), sizeof( size_t ) );

    for( auto const& row : matrix->data )
    {
        file.write( reinterpret_cast<const char*>( row.data() ), sizeof( double ) * row.size() );
    }

    file.close();
}
