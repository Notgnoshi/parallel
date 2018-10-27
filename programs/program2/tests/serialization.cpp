#include "serialization.h"
#include "matrix.h"

void SerializationTest::SimpleSmall()
{
    std::vector<std::vector<double>> v = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };
    auto matrix = std::make_shared<Matrix_t>( 3, 3 );
    matrix->data = v;

    double i = 1;
    for( auto const& row : matrix->data )
    {
        for( auto const& elem : row )
        {
            CPPUNIT_ASSERT( elem == i );
            i++;
        }
    }

    Serialize( matrix, "/tmp/SimpleSmall.mat" );

    std::shared_ptr<Matrix_t> deser = Deserialize( "/tmp/SimpleSmall.mat" );

    i = 1;
    for( auto const& row : deser->data )
    {
        for( auto const& elem : row )
        {
            CPPUNIT_ASSERT( elem == i );
            i++;
        }
    }

    CPPUNIT_ASSERT( deser->data == v );
    CPPUNIT_ASSERT( *matrix == *deser );
}
