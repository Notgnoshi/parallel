#include "serialization.h"
#include "matrix.h"

void SerializationTest::SimpleSmall()
{
    // Create a small test matrix.
    Matrix_t matrix( 3, 3 );
    for( size_t i = 0; i < matrix.elements; ++i )
    {
        matrix.data[i] = static_cast<double>( i );
    }

    // Verify that the test matrix has been created as expected.
    double v = 0;
    for( size_t i = 0; i < matrix.rows; ++i )
    {
        for( size_t j = 0; j < matrix.cols; ++j )
        {
            CPPUNIT_ASSERT( matrix( i, j ) == v );
            ++v;
        }
    }

    // Serialize the matrix to a file.
    matrix.Serialize( "/tmp/SimpleSmall.mat" );

    // Deserialize the file into a new matrix.
    Matrix_t deser( "/tmp/SimpleSmall.mat" );

    // Verify that the new matrix has the right values in its data array.
    for( size_t i = 0; i < deser.elements; ++i )
    {
        CPPUNIT_ASSERT( deser.data[i] == static_cast<double>( i ) );
    }

    // Verify that the new matrix has the right values in the right places.
    v = 0;
    for( size_t i = 0; i < deser.rows; ++i )
    {
        for( size_t j = 0; j < deser.cols; ++j )
        {
            CPPUNIT_ASSERT( deser( i, j ) == v );
            ++v;
        }
    }

    // Verify that the matrices are equal.
    CPPUNIT_ASSERT( matrix == deser );
}

void SerializationTest::SimplePack()
{
    Matrix_t matrix( 4, 4 );
    for( size_t i = 0; i < matrix.elements; ++i )
    {
        matrix.data[i] = 1;
    }

    // Serialize the matrix somewhere safe to examine the hexdump if necessary.
    matrix.Serialize( "/tmp/SimplePack.4x4_ones.mat" );

    //! @note This path is relative to the directory the executable is ran from,
    //! not the executable itself.
    Matrix_t fxf( "./matrices/4x4_ones.mat" );

    CPPUNIT_ASSERT( matrix == fxf );
}

void SerializationTest::SerializationOrder()
{
    Matrix_t matrix( 4, 4 );
    for( size_t i = 0; i < matrix.elements; ++i )
    {
        matrix.data[i] = static_cast<double>( i + 1 );
    }

    Matrix_t expected( "./matrices/4x4_seq.mat" );

    CPPUNIT_ASSERT( matrix == expected );

    CPPUNIT_ASSERT( expected( 0, 0 ) == 1 );
    CPPUNIT_ASSERT( expected( 3, 3 ) == 16 );

    for( size_t r = 0; r < matrix.rows; ++r )
    {
        for( size_t c = 0; c < matrix.cols; ++c )
        {
            CPPUNIT_ASSERT( matrix( r, c ) == expected( r, c ) );
        }
    }

    matrix.Serialize( "/tmp/SerializationOrder_seq.mat" );

    Matrix_t expected2( "/tmp/SerializationOrder_seq.mat" );

    CPPUNIT_ASSERT( expected == expected2 );
    CPPUNIT_ASSERT( matrix == expected2 );
}
