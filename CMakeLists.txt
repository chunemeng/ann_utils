cmake_minimum_required(VERSION 3.15)
project(ann CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(
        include
        src
)

add_subdirectory(src)

