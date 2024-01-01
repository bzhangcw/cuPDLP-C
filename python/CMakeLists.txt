# @author: C.Z <chuwen@shanshu>
# @note: add SWIG interface with anaconda python
#   to cupdlp to enable a matrix-vector interface
# settings
set(ANACONDA_PYTHON_VERBOSE ON)
set(CPPDP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cc/dp")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
# packages
find_package(SWIG REQUIRED)
include_directories(${SWIG_INCLUDE_DIR})
include(${SWIG_USE_FILE})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include(FindConda)

message(NOTICE "PYTHON INCLUDE: ${PYTHON_INCLUDE_DIRS}")
message(NOTICE "CPP-DP INCLUDE: ${CPPDP_INCLUDE_DIR}")


# SWIG PYTHON LIB
set_property(SOURCE pdlp.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE pdlp.i PROPERTY SWIG_MODULE_NAME pypdlp)

swig_add_library(pypdlp
        TYPE SHARED
        LANGUAGE python
        SOURCES pdlp.i)

target_include_directories(
        pypdlp
        PRIVATE
        ${CPPDP_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        ${PROJECT_BINARY_DIR}
)

target_link_libraries(
        pypdlp
        ${PYTHON_LIBRARY}
        cupdlp
)

set_property(TARGET pypdlp PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

# note: macOS is APPLE and also UNIX !
if(APPLE)
    message(NOTICE "Add apple properties")
    set_target_properties(pypdlp PROPERTIES
            SUFFIX ".so"
            INSTALL_RPATH "@loader_path;@loader_path./"
            )
    set_property(TARGET pypdlp APPEND PROPERTY
            LINK_FLAGS "-flat_namespace -undefined suppress"
            )
elseif(UNIX)
    message(NOTICE "Add unix properties")
    set_target_properties(pypdlp PROPERTIES
            INSTALL_RPATH "$ORIGIN:$ORIGIN/"
            )
endif()

add_custom_target(
        deploy
        COMMAND cp -r "${CMAKE_BINARY_DIR}/lib/*" ${PROJECT_SOURCE_DIR}/../pypdlproute
        COMMAND cp -r "${CMAKE_BINARY_DIR}/python/*.py" ${PROJECT_SOURCE_DIR}/../pypdlproute/
        DEPENDS pypdlp
)

# Variable PYTHON_LIBRARIES can contains keyword `optimized`
# which won't be interpreted inside a generator expression.
# i.e. we can't use: $<$<PLATFORM_ID:Windows>:${PYTHON_LIBRARIES}>
# see: https://cmake.org/cmake/help/git-stage/command/target_link_libraries.html#command:target_link_libraries

