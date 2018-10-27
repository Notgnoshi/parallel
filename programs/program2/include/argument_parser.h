#pragma once

#include <string>
#include <vector>

/**
 * @brief Parses commandline arguments for this program.
 */
class ArgumentParser
{
public:
    //! @brief The different operations this program is capable of.
    enum Operation_e
    {
        OPERATION_VECTOR_MULTIPLICATION, //!< Multiply left matrix by right vector.
        OPERATION_ADDITION,              //!< Add left matrix to right matrix.
    };

    //! @brief The CUDA kernels to pick from when performing the chosen operation.
    enum Kernel_e
    {
        KERNEL_DEFAULT = 0, //!< Use the default kernel for the chosen operation.
        KERNEL_CPU = 1,     //!< Do not use a CUDA kernel; perform all computation on the CPU
    };

    //! @brief A nice wrapper for the possible commandline arguments.
    struct Args_t
    {
        //! @brief The path to the left operand.
        std::string left_input = "";
        //! @brief The path to the right operand.
        std::string right_input = "";
        //! @brief Whether or not to output the results to the given file.
        bool output = false;
        //! @brief The output filename.
        std::string output_file = "";
        //! @brief The chosen operation.
        Operation_e operation = OPERATION_VECTOR_MULTIPLICATION;
        //! @brief The kernel to use for the operation.
        Kernel_e kernel = KERNEL_DEFAULT;
    };

    /**
     * @brief Construct a new Argument Parser object
     *
     * @param argc The usual number of commandline arguments.
     * @param argv The usual array of arguments.
     */
    ArgumentParser( int argc, const char** argv );

    /**
     * @brief Parse the commandline arguments.
     *
     * @note Will always return a valid Args_t object. If any invalid arguments,
     * or combination of arguments are given, this function will exit the program,
     * setting the exit status if need be.
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

    /**
     * @brief Determine if the given filename exists.
     *
     * @returns true if the file exists, false otherwise.
     */
    static bool FileExists( const std::string& filename );
};
