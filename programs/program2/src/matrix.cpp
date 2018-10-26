#include "matrix.h"
#include <cstddef>
#include <cstdint>
#include <fstream>

Matrix_t Deserialize( std::string filename )
{
    std::ifstream file( filename, std::ios::binary );

    size_t rows = 0;
    size_t cols = 0;

    file.read( reinterpret_cast<char*>( &rows ), sizeof( size_t ) );
    file.read( reinterpret_cast<char*>( &cols ), sizeof( size_t ) );

    Matrix_t matrix( rows, cols );

    file.read( reinterpret_cast<char*>( &matrix ), sizeof( matrix ) );

    //! @todo Return a shared_ptr to avoid copying...
    return matrix;
}

void Serialize( Matrix_t matrix, std::string filename )
{
    std::ofstream file( filename, std::ios::binary | std::ios::trunc );

    file.write( reinterpret_cast<const char*>( &matrix ), sizeof( matrix ) );
}
