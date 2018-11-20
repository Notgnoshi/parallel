#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief The different solution strategies to pick from.
 */
enum Strategy_e
{
    STRATEGY_BEGIN,
    STRATEGY_SERIAL = 1,           //!< A slow serial solution.
    STRATEGY_SHARED = 2,           //!< A shared memory solution.
    STRATEGY_DISTRIBUTED = 3,      //!< A distributed memory solution.
    STRATEGY_DISTRIBUTED_CUDA = 4, //!< A distributed CUDA solution.
    STRATEGY_END,
};

/**
 * @brief Parse commandline arguments for this program.
 */
class ArgumentParser
{
public:
    /**
     * @brief A tidy structure to hold the parsed commandline arguments.
     */
    struct Args_t
    {
        size_t n = 0;                          //!< The problem size to solve.
        Strategy_e strategy = STRATEGY_SERIAL; //!< The solution strategy to use.
        std::string output = "";               //!< If outputting to a file, which file.
        bool time = false;                     //!< Whether to time the execution.
    };

    /**
     * @brief Construct a new Argument Parser object
     *
     * @param argc The usual number of commandline arguments.
     * @param argv The usual array of arguments.
     */
    ArgumentParser( int argc, char** argv );

    /**
     * @brief Parse the commandline arguments.
     *
     * @note Will always return a valid Args_t object. If any invalid arguments,
     * or combination of arguments are given, this function will exit the program,
     * setting the proper exit status.
     *
     * @returns A valid Args_t object containing the parsed arguments.
     */
    Args_t ParseArgs();

    /**
     * @brief Print the program usage statement.
     */
    void Usage();

private:
    int argc;
    std::vector<std::string> argv;
};
