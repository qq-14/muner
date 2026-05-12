cmake_minimum_required(VERSION 3.10)
project(TrafficLightRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_definitions(-DTEMPLATE_DIR="${CMAKE_SOURCE_DIR}/Tamplate")
add_definitions(-DVIDEO_PATH="${CMAKE_SOURCE_DIR}/Tamplate/traffic.mp4")

file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.hpp)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
