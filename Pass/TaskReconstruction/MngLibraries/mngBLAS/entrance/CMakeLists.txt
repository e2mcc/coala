# ------------------------------------------------------------------
# 添加源文件
# ------------------------------------------------------------------
file(GLOB_RECURSE SOURCE "src/*.cpp")

# ----------------------------------------------------------
# Coala Task Reconstruction Manage BLAS (coalatrmblas)
# 生成 libcoalatrmblas.a
# ----------------------------------------------------------
add_library(coalatrmblas ${SOURCE})


# ------------------------------------------------------------------
# 添加头文件
# ------------------------------------------------------------------
target_include_directories(coalatrmblas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coalatrmblas PRIVATE ${TASK_RECONSTRUCTION_DIR}/Cornerstone/include)
target_include_directories(coalatrmblas PRIVATE ${TASK_RECONSTRUCTION_DIR}/MngMemory/include)

# 添加头文件：level3/gemm
target_include_directories(coalatrmblas PUBLIC ${MNG_BLAS_DIR}/level3/gemm)


# ------------------------------------------------------------------
# 添加依赖库
# ------------------------------------------------------------------
target_link_libraries(coalatrmblas coalatrcs)
target_link_libraries(coalatrmblas coalatrmm)


# 添加依赖库：level3
target_link_libraries(coalatrmblas coalatrmblasl3)