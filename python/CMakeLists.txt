cmake_minimum_required(VERSION 3.2) # 指定最低的CMake版本要求
project(needle C CXX) # 项目名称指定为needle, 使用编程语言c c++

# find correct version of Python
# 执行了外部命令找到python的根目录路径
# 这里的python 可以替换为conda环境也行
# 这里找到的python 执行器决定了python环境的使用，即后面的package和include地址
execute_process(COMMAND python -c "import sys; print(sys.executable)" 
    OUTPUT_VARIABLE Python_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE) # 用于删除输出结果末尾的空白字符
# 找到 Python 的 Development（开发）和 Interpreter（解释器）组件，并要求找到它们。
# 这将用于与 Python 的开发和运行时交互。
find_package(Python COMPONENTS Development Interpreter REQUIRED)
# 将 Python 的包含目录添加到项目的包含路径中，以便 CMake 和项目可以找到 Python 的头文件
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()

