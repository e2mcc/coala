#=========================================================
# 0. CMake 基本配置
#=========================================================
cmake_minimum_required(VERSION 3.27.4)

# 设置 C 编译器 （可选, 推荐使用 gcc, 否则容易报各种头文件找不到）
set (CMAKE_C_COMPILER "/usr/bin/gcc")
# 设置 C++ 编译器 （可选, 推荐使用 g++, 否则容易报各种头文件找不到）
set (CMAKE_CXX_COMPILER "/usr/bin/g++")

# 设置 C++ 编译器的版本为 17
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ compiler version")

# 设置 CXXFlags
# -O2 表示 Optimization 2 level， 开启2级优化
# -Wall 表示 Warning all， 使编译器产生尽可能多的警告信息
# -fdiagnostics-color=always 表示 使编译器在终端的输出尽可能地带有颜色
# 带有颜色地输出可以方便人类识别
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -fdiagnostics-color=always")

#开启编译命令在生成过程中的输出,输出在compile_commands.json文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#=========================================================
# 1. 配置项目名
#=========================================================
# 设置项目名
project(Coala)

# 输出项目名
message(STATUS "The Project name: ${CMAKE_PROJECT_NAME}")


#=========================================================
# 2. 配置和验证与项目相关的 LLVM 目录
#=========================================================
# 设置有效的 LLVM 安装目录，并将其存入变量 ${LLVM_INSTALL_DIR} 中
set(LLVM_INSTALL_DIR "/home/parallels/Softwares/LLVM-17.0.6" CACHE PATH "LLVM installation directory")

# 对 LLVM 安装目录的合法性进行验证
set(LLVM_INSTALL_INCLUDE "${LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LLVM_INSTALL_INCLUDE}")
	message(FATAL_ERROR "LLVM_INSTALL_INCLUDE (${LLVM_INSTALL_INCLUDE}) is invalid.")
endif()

# 对 LLVMConfig.cmake 所在目录的合法性进行验证
set(LLVM_INSTALL_CMAKE "${LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
if(NOT EXISTS "${LLVM_INSTALL_CMAKE}")
	message(FATAL_ERROR "LLVM_INSTALL_CMAKE (${LLVM_INSTALL_CMAKE}) is invalid.")
endif()

# 添加 LLVMConfig.cmake 的路径到 CMake 的搜索路径种
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/llvm/")

# 输出信息
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_INSTALL_CMAKE}")


#=========================================================
# 3 加载 LLVM 配置
#=========================================================
# 找包
find_package(LLVM REQUIRED CONFIG)

# 设置 llvm 头文件路径
include_directories(${LLVM_INCLUDE_DIRS})

# 设置 llvm 库文件路径 (所有都在 LLVM_LIBRARY_DIRS 目录下)
link_directories(${LLVM_LIBRARY_DIRS})


# 设置 -D 选项
add_definitions(${LLVM_DEFINITIONS})

# 1. LLVM通常在没有RTTI的情况下构建.
# 2. -fPIC: 为静态库合成动态库时，code与位置无关
if(NOT LLVM_ENABLE_RTTI)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fPIC")
endif()

# 构建LLVM时设置-fvisibility-inlines-hidden，
# 如果构建项目时未使用此标志，则会触发Darwin警告（否则构建良好）。
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
if (${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG} EQUAL "1")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# 输出信息
message(STATUS "LLVM INCLUDE DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM LIBRARY DIRS: ${LLVM_LIBRARY_DIRS}")

#=========================================================
# 4. BUILD 配置
#=========================================================
# 设置build文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")#设置动态库的存放目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")#设置静态库的存放目录

# 设置 Build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE
		STRING "Build type (default Debug):" FORCE)
endif()

#=========================================================
# 5. 添加子目录
# 在最后做这一步，以便让之前所有定义和链接/包含路径都可作用于子目录
#=========================================================
add_subdirectory(Pass)
add_subdirectory(Lazy)
add_subdirectory(Allocation)