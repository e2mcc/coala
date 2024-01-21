# ----------------------------------------------------------
# Coala Runtime Memops (coalartmemops)
# 生成 libcoalartmemops.a
# ----------------------------------------------------------
file(GLOB_RECURSE SOURCE "src/*.c")

#最终生成的目标：生成.so文件:lib库名.so
add_library(coalartmemops ${SOURCE})

target_include_directories(coalartmemops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coalartmemops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../Probes/include)