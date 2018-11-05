#pragma once

#include <cstddef>
#include <string>

/**
 * @brief A struct wrapper for a matrix to make de/serialization easier.
 */
struct Matrix_t
{
    //! @brief The number of rows in the matrix.
    size_t rows = 0;
    //! @brief The number of columns in the matrix.
    size_t cols = 0;
    //! @brief The total number of elements.
    size_t elements = 0;

    //! @brief The matrix data, stored in row-major representation.
    double* data = nullptr;

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

    Matrix_t() = default;

    Matrix_t( const Matrix_t& other );
    /**
     * @brief Destroy the Matrix_t object.
     */
    ~Matrix_t();

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
     * @brief Pretty-print the matrix.
     */
    void Print() const;

    /**
     * @brief Determine if two matrices are equal.
     *
     * @param other The other matrix to compare against.
     * @returns true if the matrices are equal, and false otherwise.
     */
    bool operator==( const Matrix_t& other ) const;

    /**
     * @brief Provide a 2D interface to the 1D internal array.
     *
     * @param row
     * @param col
     * @returns The element stored at the [row][col] position.
     */
    double& operator()( size_t row, size_t col ) const;

    /**
     * @brief Provide a 2D interface to the 1D internal array.
     *
     * @param row
     * @param col
     * @returns The element stored at the [row][col] position.
     */
    double& operator()( size_t row, size_t col );

    /**
     * @brief Allocate and zero a contiguous 2D array.
     *
     * @details Actually allocate a 1D array of size rows*cols.
     *
     * @param rows The number of rows in the array.
     * @param cols The number of columns in the array.
     * @returns A single pointer to the allocated and zeroed array, or nullptr on failure.
     */
    static double* Allocate2D( size_t rows, size_t cols );
};
