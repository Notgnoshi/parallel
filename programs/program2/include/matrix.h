#pragma once

#include <memory>
#include <string>
#include <vector>

/**
 * @brief A struct wrapper for a matrix to make de/serialization easier.
 */
struct Matrix_t
{
    //! @brief The number of rows in the matrix.
    size_t rows;
    //! @brief The number of columns in the matrix.
    size_t cols;

    //! @brief The matrix data, stored in row-major representation.
    //! @todo Experiment with column-major.
    std::vector<std::vector<double>> data;

    /**
     * @brief Construct an empty Matrix_t object of the given size.
     *
     * @param rows
     * @param cols
     */
    Matrix_t( size_t rows, size_t cols ) :
        rows( rows ),
        cols( cols ),
        data( rows, std::vector<double>( cols, 0 ) )
    {
    }

    /**
     * @brief Determine if two matrices are equal.
     *
     * @param other The other matrix to compare against.
     * @returns true if the matrices are equal, and false otherwise.
     */
    bool operator==( const Matrix_t& other ) const
    {
        return this->rows == other.rows && this->cols == other.cols && this->data == other.data;
    }
};

/**
 * @brief Deserialize a matrix from a file.
 *
 * @note Assumes the file has no additional cruft.
 *
 * @param filename The file to read the matrix from.
 * @returns A shared pointer to the deserialized matrix.
 */
std::shared_ptr<Matrix_t> Deserialize( const std::string& filename );

/**
 * @brief Serializes a matrix to the given filename.
 *
 * @note This will overwrite the file if it already exists.
 *
 * @details The matrix will be saved to a file as follows.
 * * The first 8 bytes (size_t) indicate the number of rows.
 * * The second 8 bytes (size_t) indicate the number of columns in each row.
 * * The next rows * cols * 8 (double) bytes of the file is the matrix data saved in
 *   row major format.
 *
 * @param matrix   A shared pointer to the matrix to serialize.
 * @param filename The filename to save the matrix as.
 */
void Serialize( const std::shared_ptr<Matrix_t>& matrix, const std::string& filename );
