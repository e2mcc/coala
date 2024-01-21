# ------------------------------------------------------------------
# 添加源文件
# ------------------------------------------------------------------
file(GLOB_RECURSE SOURCE "gemm/*.cpp")

# ------------------------------------------------------------------
#最终生成的目标：生成.a文件:lib库名.a
# ------------------------------------------------------------------
add_library(coalatrmblasl3 ${SOURCE})

# ------------------------------------------------------------------
# 依赖的头文件
# ------------------------------------------------------------------
target_include_directories(coalatrmblasl3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gemm)
target_include_directories(coalatrmblasl3 PRIVATE ${MNG_BLAS_DIR}/entrance/include)
target_include_directories(coalatrmblasl3 PRIVATE ${TASK_RECONSTRUCTION_DIR}/Cornerstone/include)
target_include_directories(coalatrmblasl3 PRIVATE ${TASK_RECONSTRUCTION_DIR}/MngMemory/include)

# ------------------------------------------------------------------
# 添加依赖库
# ------------------------------------------------------------------
target_link_libraries(coalatrmblasl3 coalatrcs)
target_link_libraries(coalatrmblasl3 coalatrmm)