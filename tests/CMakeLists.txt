# Copyright 2022 David Feltell
# SPDX-License-Identifier: MIT
add_subdirectory(unit)
# TODO(DF): The Ruby dependency of cucumber-cpp-runner is a pain (build failures installing Gems,
# 	which are impossible to fix). Until the dependency is removed, lets not bother with Cucumber
#	tests.
#add_subdirectory(feature)
