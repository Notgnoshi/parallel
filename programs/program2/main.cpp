#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>

/**
 * @brief Main entry point for the application.
 *
 * @returns 0 to indicate success.
 */
int main()
{
    std::ofstream file( "/tmp/SimpleSmall.mat", std::ofstream::out | std::ofstream::binary | std::ofstream::trunc );

    file.close();

    return 0;
}
