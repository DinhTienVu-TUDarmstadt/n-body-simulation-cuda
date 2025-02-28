# CMake generated Testfile for 
# Source directory: D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/test_lab3
# Build directory: D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/test_lab3
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SerialTests "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/cmake-build-debug-visual-studio/bin/lab3_test.exe")
set_tests_properties(SerialTests PROPERTIES  _BACKTRACE_TRIPLES "D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/test_lab3/CMakeLists.txt;80;add_test;D:/TU Darmstadt/Semester3/Parallele Programmierung/lab_3/test_lab3/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
