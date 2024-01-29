# ------------------------------------------------------------------
# 添加源文件
# ------------------------------------------------------------------
file(GLOB_RECURSE SOURCE "gemv/src/*.c")

# ------------------------------------------------------------------
#最终生成的目标：生成.a文件:lib库名.a
# ------------------------------------------------------------------
add_library(coalablasl2 ${SOURCE})

# ------------------------------------------------------------------
# 依赖的头文件
# ------------------------------------------------------------------
# 依赖 coala_blas_gemm.h
target_include_directories(coalablasl2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gemv/include)
# 依赖 coala_blas_base.h
target_include_directories(coalablasl2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../base/include)
# 依赖 coala_probes.h
target_include_directories(coalablasl2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../Lazy/Probes/include)

# ------------------------------------------------------------------
# 添加依赖库
# ------------------------------------------------------------------
# target_link_libraries(coalablasl3 coalatrcs)
# target_link_libraries(coalablasl3 coalatrmm)