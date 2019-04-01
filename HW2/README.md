Programming Assignment I: Pthreads Programming
===========================
The purpose of this assignment is to familiarize yourself with OpenMP programming.



****
Contents
------
* [Problem Statement](#Problem-Statement)
* [Requirement](#Requirement)
* [Sumbission](#Submission) 
* [Reference](#Reference)

****
Problem Statement
------
Conjugate gradient method is an algorithm for the numerical solution of particular systems of linear equations. It is often used to solve partial differential equations, or applied on some optimization problems. You may get more information on [Wikipedia](http://en.wikipedia.org/wiki/Conjugate_gradient_method). In this assignment, you are asked to parallelize a serial implementation of the conjugate gradient method using OpenMP. The serial implementation can be downloaded at [http://www.cs.nctu.edu.tw/~ypyou/courses/PP-s19/assignments/HW2/CG.tgz](http://www.cs.nctu.edu.tw/~ypyou/courses/PP-s19/assignments/HW2/CG.tgz). It contains:
* cg.c <br>
  The implementation of the conjugate gradient method.
* globals.h <br>
  Some data definitions. ***DO NOT*** modify this file.
* common directory <br>
  Directory that contains some functions for time calculation and random numbers. ***DO NOT*** modify the files in this directory.
* bin directory <br>
  Directory that contains the executable.
* Makefile, make.common <br>
  Makefiles.
* README <br>
  The information of the program.

****
Requirement
------
In this assignment, you have to modify cg.c to improve the performance of this program (i.e.,to insert OpenMP pragmas/library routines to parallelize parts of the program). Of course, you may add any other variables or functions if necessary.

Note:
* ***DO NOT*** modify/add any output messages.
* Grading will be made based on the speedup/efficiency that your implementation would yield.

****
Submission
------
Please rename your cg.c to ***\<your-student-id\>.c*** and upload it to e-Campus system by the due date.

***Due Date: 23:59, April 15, 2019***

****
References
------
* [http://openmp.org/](http://openmp.org/)
* [https://computing.llnl.gov/tutorials/openMP/](https://computing.llnl.gov/tutorials/openMP/)


