SET(TASK_RECONSTRUCTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# ----------------------------------------------------------
# Coala Task Reconstruction(coalatr)
# 生成 libcoalatr.a 合 libcoalatr.so
# ----------------------------------------------------------

add_subdirectory(HandleTraverse)
add_subdirectory(MngLibraries)
add_subdirectory(MngMemory)
add_subdirectory(Cornerstone)

#最终生成的目标：生成.so文件:lib库名.so
add_library(coala MODULE coala_tr_entrance.cpp)


# 链接子文件夹生成的库
target_link_libraries(coala coalatrht)
target_link_libraries(coala coalatrmblas)
target_link_libraries(coala coalatrmm)
target_link_libraries(coala coalatrcs)