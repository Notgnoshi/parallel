#pragma once

#include <string>
#include <vector>

/**
 * @brief A struct wrapper for a matrix to make de/serialization easier.
 */
#pragma pack( 1 )
struct Matrix_t
{
    size_t rows;
    size_t cols;

    std::vector<std::vector<double>> matrix;

    /**
     * @brief Construct an empty Matrix_t object of the given size.
     *
     * @note This is a row-major representation.
     * @todo Experiment with column-major.
     *
     * @param rows
     * @param cols
     */
    Matrix_t( size_t rows, size_t cols ) :
        rows( rows ),
        cols( cols ),
        matrix( rows, std::vector<double>( cols, 0 ) )
    {
    }

    bool operator==( const Matrix_t& other ) const
    {
        if( this->rows != other.rows || this->cols != other.cols )
        {
            return false;
        }

        size_t r = 0;
        for( auto const& row : this->matrix )
        {
            if( row != other.matrix[r] )
            {
                return false;
            }

            r++;
        }

        return true;
    }
};

/**
 * @brief Deserialize a matrix from a file.
 *
 * @note Assumes the file has no additional cruft.
 *
 * @param filename The file to read the matrix from.
 * @returns The deserialized matrix.
 */
Matrix_t Deserialize( std::string filename );

/**
 * @brief Serializes a matrix to the given filename.
 *
 * @note This will overwrite the file if it already exists.
 *
 * @details The matrix will be saved to a file as follows.
 * * The first 8 bytes (size_t) indicate the number of rows.
 * * The second 8 bytes (size_t) indicate the number of columns in each row.
 * * The next rows * cols * 8 bytes of the file is the matrix data saved in
 *   row major format.
 *
 * @param matrix   The matrix to serialize.
 * @param filename The filename to save the matrix as.
 */
void Serialize( Matrix_t matrix, std::string filename );
