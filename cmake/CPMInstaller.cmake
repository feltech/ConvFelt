include_guard(GLOBAL)

set(
	convfelt_DEPENDENCY_INSTALL_CACHE_DIR ${PROJECT_BINARY_DIR}/_deps/dist
	CACHE PATH "Location to install any missing dependencies prior to the build process"
)

#------------------------------------------------------------
# Install CPM package manager

set(CPM_DOWNLOAD_VERSION 0.34.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
	message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
	file(DOWNLOAD
		https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
		${CPM_DOWNLOAD_LOCATION}
	)
endif ()

include(${CPM_DOWNLOAD_LOCATION})

function(convfelt_cpm_install_package)
	cmake_parse_arguments(
		PARSE_ARGV 0
		args
		""
		"NAME;GIT_TAG;GITHUB_REPOSITORY;BUILD_TYPE"
		"FIND_PACKAGE_OPTIONS;CMAKE_OPTIONS"
	)

	find_package(
		${args_NAME} CONFIG
		PATHS ${convfelt_DEPENDENCY_INSTALL_CACHE_DIR}
		${args_FIND_PACKAGE_OPTIONS}
	)
	if (NOT DEFINED ${args_NAME}_CONFIG)
		message(STATUS "${args_NAME} not found, downloading...")

		CPMAddPackage(
			NAME ${args_NAME}
			SYSTEM YES
			EXCLUDE_FROM_ALL YES
			DOWNLOAD_ONLY YES
			GIT_TAG ${args_GIT_TAG}
			GITHUB_REPOSITORY ${args_GITHUB_REPOSITORY}
		)

		if (NOT args_BUILD_TYPE)
			set(args_BUILD_TYPE Release)
		endif ()

		if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
			# Frustratingly, need to keep a list of warnings to disable, for annoying projects
			# that force `-Werror` even for consumers (e.g. Cucumber-Cpp)
			# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329 (affects Cucumber-Cpp)
			list(APPEND _warnings_to_disable
				-D_CRT_SECURE_NO_WARNINGS
				-Wno-unknown-warning-option
				-Wno-restrict
				-Wno-deprecated
				-Wno-unused-command-line-argument
			)
			list(JOIN _warnings_to_disable " " _warnings_to_disable)
			list(APPEND args_CMAKE_OPTIONS "-DCMAKE_CXX_FLAGS=${_warnings_to_disable}")
		endif ()

		if (BUILD_SHARED_LIBS)
			# In case we're linking a static library into a shared library.
			list(APPEND args_CMAKE_OPTIONS -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
		elseif (DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
			list(APPEND args_CMAKE_OPTIONS -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE})
		endif ()

		if (CMAKE_PREFIX_PATH)
			string(REPLACE ";" "$<SEMICOLON>" _prefix_path "${CMAKE_PREFIX_PATH}")
			list(APPEND args_CMAKE_OPTIONS "-DCMAKE_PREFIX_PATH=${_prefix_path}")
		endif ()

		if (CMAKE_TOOLCHAIN_FILE)
			# Pass along any toolchain file providing e.g. 3rd party libs via Conan.
			list(APPEND args_CMAKE_OPTIONS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
		endif ()

		list(PREPEND args_CMAKE_OPTIONS
			-S "${${args_NAME}_SOURCE_DIR}"
			-B "${${args_NAME}_BINARY_DIR}"
			-G "${CMAKE_GENERATOR}"
			-DCMAKE_BUILD_TYPE=${args_BUILD_TYPE}
			--compile-no-warning-as-error
			"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
			# For MSVC and Conan v2
			-DCMAKE_POLICY_DEFAULT_CMP0091=NEW
			# Allow `PackageName_ROOT` hint for find_package calls.
			-DCMAKE_POLICY_DEFAULT_CMP0074=NEW
			#			-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
			-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
			#			-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
			#			-DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
			# TODO(DF): <$IF:$<BOOL:CMAKE_FIND_PACKAGE_PREFER_CONFIG ?
			-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=TRUE
			-DCMAKE_VERBOSE_MAKEFILE=ON
		)

		message(TRACE "${args_NAME} options: ${args_CMAKE_OPTIONS}")

		execute_process(
			COMMAND ${CMAKE_COMMAND}
			${args_CMAKE_OPTIONS}
			COMMAND_ERROR_IS_FATAL ANY
		)
		execute_process(
			COMMAND ${CMAKE_COMMAND}
			--build ${${args_NAME}_BINARY_DIR} --config ${args_BUILD_TYPE} --parallel
			COMMAND_ERROR_IS_FATAL ANY
		)
		execute_process(
			COMMAND ${CMAKE_COMMAND}
			--install ${${args_NAME}_BINARY_DIR}
			--prefix ${convfelt_DEPENDENCY_INSTALL_CACHE_DIR}
			--config ${args_BUILD_TYPE}
			COMMAND_ERROR_IS_FATAL ANY
		)
		find_package(
			${args_NAME} CONFIG REQUIRED
			PATHS ${convfelt_DEPENDENCY_INSTALL_CACHE_DIR}
			${args_FIND_PACKAGE_OPTIONS}
		)
	endif ()
endfunction()
