cmake_minimum_required(VERSION 3.16)
project(apriltag VERSION 3.4.5 LANGUAGES C CXX)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build example executables" OFF)
option(ASAN "Use AddressSanitizer for debug builds to detect memory issues" OFF)

if (ASAN)
    set(ASAN_FLAGS "\
        -fsanitize=address \
        -fsanitize=bool \
        -fsanitize=bounds \
        -fsanitize=enum \
        -fsanitize=float-cast-overflow \
        -fsanitize=float-divide-by-zero \
        -fsanitize=nonnull-attribute \
        -fsanitize=returns-nonnull-attribute \
        -fsanitize=signed-integer-overflow \
        -fsanitize=undefined \
        -fsanitize=vla-bound \
        -fno-sanitize=alignment \
        -fsanitize=leak \
        -fsanitize=object-size \
    ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
endif()

# Set a default build type if none was specified
set(default_build_type "Release")

SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE  STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS  "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra)
    add_compile_options(-Wpedantic)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-strict-prototypes
            -Wno-static-in-inline
        )
    endif()
    add_compile_options(-Wno-shift-negative-value)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
      set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    else()
      add_compile_options(-Werror)
    endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_C_SIMULATE_ID MATCHES "MSVC")
    # error: 'strdup' is deprecated: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name: _strdup.
    # "strdup" is standard since C23
    add_definitions(-D _CRT_NONSTDC_NO_DEPRECATE)
    # ignore "'fopen' is deprecated" and "'strncpy' is deprecated" warnings
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

aux_source_directory(common COMMON_SRC)
set(APRILTAG_SRCS apriltag.c apriltag_pose.c apriltag_quad_thresh.c)

# Library
file(GLOB TAG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tag*.c)
add_library(${PROJECT_NAME} ${APRILTAG_SRCS} ${COMMON_SRC} ${TAG_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT APPLE AND NOT CMAKE_C_SIMULATE_ID MATCHES "MSVC")
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-z,relro,-z,now,-z,defs")
endif()

if (MSVC)
    add_compile_definitions("_CRT_SECURE_NO_WARNINGS")
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)
endif()

if (UNIX)
    target_link_libraries(${PROJECT_NAME} PUBLIC m)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 3 VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")
set_property(TARGET ${PROJECT_NAME} PROPERTY C_STANDARD 99)


include(GNUInstallDirs)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/apriltag")


# Python wrapper using nanobind
option(BUILD_PYTHON_WRAPPER "Builds Python wrapper" ON)

if(BUILD_PYTHON_WRAPPER)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Detect nanobind
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(apriltag_py STABLE_ABI ${CMAKE_CURRENT_SOURCE_DIR}/apriltag_nanobind.cpp)
    target_link_libraries(apriltag_py PRIVATE apriltag)
    target_compile_features(apriltag_py PRIVATE cxx_std_20)
    set_target_properties(apriltag_py PROPERTIES OUTPUT_NAME apriltag)

    if(NOT SKBUILD)
        set(PY_DEST ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/)
    else()
        set(PY_DEST .)
    endif()
    install(TARGETS apriltag_py LIBRARY DESTINATION ${PY_DEST})
endif()

# Examples
if (BUILD_EXAMPLES)
    # apriltag_demo
    add_executable(apriltag_demo example/apriltag_demo.c)
    target_link_libraries(apriltag_demo ${PROJECT_NAME})

    # opencv_demo
    set(_OpenCV_REQUIRED_COMPONENTS core imgproc videoio highgui)
    find_package(OpenCV COMPONENTS ${_OpenCV_REQUIRED_COMPONENTS} QUIET CONFIG)
    if(OpenCV_FOUND)
        enable_language(CXX)
        # NB: contrib required for TickMeter in OpenCV 2.4. This is only required for 16.04 backwards compatibility and can be removed in the future.
        #     If we add it to the find_package initially, the demo won't build for newer OpenCV versions
        if(OpenCV_VERSION VERSION_LESS "3.0.0")
            list(APPEND _OpenCV_REQUIRED_COMPONENTS contrib)
            find_package(OpenCV COMPONENTS ${_OpenCV_REQUIRED_COMPONENTS} CONFIG)
        endif()

        add_executable(opencv_demo example/opencv_demo.cc)
        target_link_libraries(opencv_demo apriltag ${OpenCV_LIBRARIES})
        set_target_properties(opencv_demo PROPERTIES CXX_STANDARD 11)
        install(TARGETS opencv_demo RUNTIME DESTINATION bin)
    else()
        message(STATUS "OpenCV not found: Not building demo")
    endif(OpenCV_FOUND)

    # install example programs
    install(TARGETS apriltag_demo RUNTIME DESTINATION bin)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
