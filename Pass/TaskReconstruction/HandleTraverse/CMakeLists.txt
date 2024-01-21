# ----------------------------------------------------------
# Coala Task Reconstruction Handle Traverse (coalatrht)
# 生成 libcoalatrht.a
# ----------------------------------------------------------

file(GLOB_RECURSE SOURCE "src/*.cpp")

#最终生成的目标：生成.so文件:lib库名.so
add_library(coalatrht ${SOURCE})



# 说明依赖头文件
target_include_directories(coalatrht PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coalatrht PRIVATE "../Cornerstone/include")


# 说明依赖库
target_link_libraries(coalatrht coalatrcs)