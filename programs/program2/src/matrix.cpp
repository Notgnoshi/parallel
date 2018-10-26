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
    std::ofstream file( "/tmp/SimpleSmall.mat", std::ofstream::out | std::ofstream::binary | std::ofstream::trunc );

    file.close();

    (void)matrix;
    (void)filename;

    // file << matrix.rows;
    // file << matrix.cols;

    // for( auto const& row : matrix.matrix )
    // {
    //     for( auto const& elem : row )
    //     {
    //         file << elem;
    //     }
    // }

    // file.close();
}
