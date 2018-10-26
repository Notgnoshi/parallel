#include "serialization.h"
#include "matrix.h"

void SerializationTest::SimpleSmall()
{
    // Zero initializes the vectors.
    Matrix_t matrix(3, 3);

    for( auto const& row : matrix.matrix )
    {
        for( auto const& elem : row )
        {
            CPPUNIT_ASSERT( elem == 0.0 );
        }
    }

    Serialize( matrix, "/tmp/SimpleSmall.mat" );

    // Matrix_t deser = Deserialize( "/tmp/SimpleSmall.mat" );

    // CPPUNIT_ASSERT( matrix == deser );
}
