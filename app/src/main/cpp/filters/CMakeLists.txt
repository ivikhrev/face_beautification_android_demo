add_library(
        # Sets the name of the library.
        filters
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        filters.cpp)
target_link_libraries(filters ${OpenCV_LIBRARIES})