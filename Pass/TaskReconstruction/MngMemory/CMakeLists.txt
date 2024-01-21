file(GLOB_RECURSE SOURCE "src/*.cpp")

#最终生成的目标：生成.so文件:lib库名.so
add_library(coalatrmm ${SOURCE})

target_include_directories(coalatrmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coalatrmm PRIVATE "../Cornerstone/include")



# 说明依赖库
target_link_libraries(coalatrmm coalatrcs)