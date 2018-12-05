#include "testsuite.h"
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

/**
 * @brief A global pointer to the number of commandline arguments.
 */
int* GLOBAL_ARGC;
/**
 * @brief A global pointer to the array of commandline arguments.
 */
const char*** GLOBAL_ARGV;

int main( int argc, const char** argv )
{
    // I want to cry. MPI requires initialization, which requires the commandline
    // arguments. There is not a way to pass (in CppUnit) parameters to individual
    // tests.
    GLOBAL_ARGC = &argc;
    GLOBAL_ARGV = &argv;

    // Get the top level suite from the registry
    Test* suite = CppUnit::TestFactoryRegistry::getRegistry().makeTest();

    // Adds the test to the list of test to run
    TextUi::TestRunner runner;
    runner.addTest( suite );

    // Change the default outputter to a compiler error format outputter
    runner.setOutputter( new CppUnit::CompilerOutputter( &runner.result(),
                                                         std::cerr ) );
    // Run the tests.
    bool wasSuccessful = runner.run();

    // Return error code 1 if the one of test failed.
    return wasSuccessful ? 0 : 1;
}
