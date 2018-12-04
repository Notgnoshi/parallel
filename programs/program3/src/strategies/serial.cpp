#include "strategies/serial.h"

SerialStrategy::SerialStrategy( std::string output, bool time ) :
    output( output ),
    time( time )
{
}

void SerialStrategy::Run( ArgumentParser::Args_t args )
{
    ArgumentParser::Summarize( args );
}
