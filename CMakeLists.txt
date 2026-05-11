cmake_minimum_required(VERSION 3.8)

project(project)

find_package(OpenCV REQUIRED)

set(SRC /src/*.cpp)
include(/include/*.hpp)

add_executable(project SRC)