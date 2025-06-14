cmake_minimum_required(VERSION 3.7)

set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)
SET(VCPKG_BOOTSTRAP_OPTIONS "-disableMetrics")
SET(X_VCPKG_APPLOCAL_DEPS_INSTALL ON)

project(opencl-app)

find_package(OpenCL REQUIRED)

# src files
file(GLOB_RECURSE all_SRC
	"src/*.cpp"
)

add_executable(opencl-app
	main.cpp
	${all_SRC}
)
target_link_libraries(opencl-app PRIVATE OpenCL::OpenCL)

# Always-run copy target
add_custom_target(CopyKernels ALL
	COMMENT "Force-copying OpenCL kernels"
	VERBATIM
)

# Find and copy .cl files
file(GLOB_RECURSE CL_FILES "${CMAKE_SOURCE_DIR}/*.cl")
list(FILTER CL_FILES EXCLUDE REGEX "${CMAKE_BINARY_DIR}")

foreach(CL_FILE IN LISTS CL_FILES)
	file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}" "${CL_FILE}")
	
	# Add to both targets
	add_custom_command(
		TARGET CopyKernels
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CL_FILE}"
		"${CMAKE_BINARY_DIR}/${REL_PATH}"
		COMMENT "Copying ${REL_PATH}"
		VERBATIM
	)
	
	add_custom_command(
		TARGET opencl-app POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CL_FILE}"
		"${CMAKE_BINARY_DIR}/${REL_PATH}"
		COMMENT "Copying ${REL_PATH}"
		VERBATIM
	)
endforeach()

# Ensure copy happens before main build
add_dependencies(opencl-app CopyKernels)
