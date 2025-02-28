# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-src"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-build"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/tmp"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/src"
  "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/_deps/cli11-subbuild/cli11-populate-prefix/src/cli11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
