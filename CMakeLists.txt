cmake_minimum_required(VERSION 3.10)

project(TrafficLightRecognition)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 路径
# Windows交叉编译时：
# cmake 会从这里找 Windows 版 OpenCV
set(OpenCV_DIR
    ${CMAKE_SOURCE_DIR}/3rdparty/opencv-win
)

find_package(OpenCV REQUIRED)

# 宏定义
add_definitions(-DTEMPLATE_DIR="${CMAKE_SOURCE_DIR}/Template")
add_definitions(-DVIDEO_PATH="${CMAKE_SOURCE_DIR}/Template/traffic.mp4")

# 源文件
file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.hpp)

# 可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 头文件
target_include_directories(${PROJECT_NAME}
PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

# Windows 静态运行库
target_link_libraries(${PROJECT_NAME}
    -static
    -static-libgcc
    -static-libstdc++
)