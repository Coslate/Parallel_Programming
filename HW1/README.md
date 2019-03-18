Programming Assignment I: Pthreads Programming
===========================
The purpose of this assignment is to familiarize yourself with Pthreads programming.


****
Contents
------
* [Problem Statement](#Problem-Statement)
* [Requirement](#Requirement)
* [Grading Policies](#Grading-Policies)
* [Evaluation Platform](#Evaluation-Platform)
* [Sumbission](#Submission) 
* [Reference](#Reference)

****
Problem Statement
------
Suppose we toss darts randomly at a square dartboard, whose bullseye is at the origin, and whose sides are two feet in length. Suppose also that there is a circle inscribed in the square dartboard. The radius of the circle is one foot, and its area is π square feet. If the points that are hit by the darts are uniformly distributed (and we always hit the square), then the number of darts that hit inside the circle should approximately satisfy the equation: 

<img src='https://latex.codecogs.com/gif.latex?\frac{\mathbf{number\;&space;of\;&space;circles}}{\mathbf{total\;number\;of\;tosses}}=&space;\frac{\pi}{4}'/>

since the ratio of the area of the circle to the area of the square is π/4.
We can use this formula to estimate the value of π with a random number generator:
'''
number_in_circle = 0;
for ( toss = 0; toss < number_of_tosses ; toss ++) {
    x = random double between -1 and 1;
    y = random double between -1 and 1;
    distance_squared = x * x + y * y ;
    if ( distance_squared <= 1)
        number_in_circle ++;
    }
pi_estimate = 4* number_in_circle /(( double ) number_of_tosses ) ;
'''

This is called a “Monte Carlo” method, since it uses randomness (the dart tosses). Write a Pthreads program that uses a Monte Carlo method to estimate π. The main thread should read in the total number of tosses and print the estimate. You may want to use long long ints for the number of hits in the circle and the number of tosses, since both may have to be very large to get a reasonable estimate of π.
