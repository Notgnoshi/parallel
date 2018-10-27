#pragma once

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
     * @brief Construct a zeroed Matrix_t object of the given size.
     *
     * @param rows
     * @param cols
     */
    Matrix_t( size_t rows, size_t cols );

    /**
     * @brief Deserialize a Matrix_t from a file.
     *
     * @param filename The file to deserialize the matrix from.
     */
    Matrix_t( const std::string& filename );

    /**
     * @brief Serialize a Matrix_t to a file.
     *
     * @note This will overwrite the file if it already exists.
     *
     * @details The matrix will be saved to a file as follows.
     * * The first 8 bytes (size_t) indicate the number of rows.
     * * The second 8 bytes (size_t) indicate the number of columns in each row.
     * * The next rows * cols * 8 (double) bytes of the file is the matrix data saved in
     *   row major format.
     *
     * @param filename The file to save the serialized matrix to.
     */
    void Serialize( const std::string& filename );

    /**
     * @brief Determine if two matrices are equal.
     *
     * @param other The other matrix to compare against.
     * @returns true if the matrices are equal, and false otherwise.
     */
    bool operator==( const Matrix_t& other ) const;
};
