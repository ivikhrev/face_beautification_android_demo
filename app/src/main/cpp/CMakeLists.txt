cmake_minimum_required(VERSION 3.22.1)
set (CMAKE_CXX_STANDARD 17)
project("processing_lib")

add_library( # Sets the name of the library.
        processing_lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)