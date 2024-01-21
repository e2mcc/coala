# ----------------------------------------------------------
# Coala Task Reconstruction Cornerstone (coalatrcs)
# 生成 libcoalatrcs.a
# ----------------------------------------------------------

file(GLOB_RECURSE SOURCE "src/*.cpp")

#最终生成的目标：生成.so文件:lib库名.so
add_library(coalatrcs ${SOURCE})

target_include_directories(coalatrcs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
