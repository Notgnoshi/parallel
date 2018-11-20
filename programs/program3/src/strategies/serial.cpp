#include "strategies/serial.h"

SerialStrategy::SerialStrategy( std::string output, bool time ) :
    output( output ),
    time( time )
{
}

size_t SerialStrategy::Run( size_t n, bool output )
{
    (void)output;
    return n;
}
