cmake_minimum_required(VERSION 3.22.1)
set (CMAKE_CXX_STANDARD 17)
project("filters")

find_package( OpenCV REQUIRED )

add_library( # Sets the name of the library.
        filters
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        filters.cpp)
target_link_libraries(filters ${OpenCV_LIBRARIES})