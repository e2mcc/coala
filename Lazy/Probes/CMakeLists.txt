# ----------------------------------------------------------
# Coala Runtime Probes (coalartprobes)
# 生成 libcoalartprobes.a
# ----------------------------------------------------------
file(GLOB_RECURSE SOURCE "src/*.c")

#最终生成的目标：生成.so文件:lib库名.so
add_library(coalartprobes ${SOURCE})

target_include_directories(coalartprobes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)