#include "common.h"
#include <fstream>

void AppendBlock( std::vector<std::vector<uint8_t>> block, std::string filename )
{
    std::ofstream fout( filename, std::ios::app );
    // Output to the file in blocks to save costly I/O.
    std::string formatted_block;

    for( auto&& solution : block )
    {
        for( auto&& row : solution )
        {
            formatted_block += std::to_string( row ) + " ";
        }
        formatted_block += '\n';
    }

    fout << formatted_block;
}

void PrintSolution( std::vector<uint8_t> solution )
{
    for( auto&& q : solution )
    {
        std::cout << static_cast<int>( q ) << " ";
    }
    std::cout << std::endl;
}
