""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
import numpy as np
from statistics import mean 
from time import perf_counter as pc
from functools import reduce
#from numba import njit
import concurrent.futures as future

# Exc1
def approximate_pi(n):
    # n is the number of points
    print("Number of points:", n)

    points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)]

    n_c = []
    n_s = []
    for x, y in points:
        if x**2 + y**2 <= 1:     # Points inside circle
            n_c.append((x, y))
        else:
            n_s.append((x, y))
    pi_approx = 4 * len(n_c) / len(points)   # n = len(points)
    
    print(pi_approx)

    # Create plot of points inside circle as red dots and outside as blue dots
    x_c = [p[0] for p in n_c]
    y_c = [p[1] for p in n_c]
    x_s = [p[0] for p in n_s]
    y_s = [p[1] for p in n_s]

    plt.scatter(x_c, y_c, color="red", s=0.3)
    plt.scatter(x_s, y_s, color="blue", s=0.3)
    plt.xticks(np.arange(-1, 1.25, 0.25))     # To have 0.25 as step size on both axes
    plt.yticks(np.arange(-1, 1.25, 0.25))     # -//-
    plt.axis("equal")
    plt.savefig(f"pi_approx_{n}.png")

    return pi_approx

# Exc2, approximation of volume of d dimensional sphere
def sphere_volume(n, d, r=1): 
    # n is the number of points
    # d is the number of dimensions of the sphere

    rng = np.random.default_rng()   # Create a random generator using numpy

    points = rng.uniform(-r, r, size=(n, d))  # Create a matrix containing the random points

    # Create a list of 1:s for every point inside the sphere
    n_d = [1 for point in points if sum(map(lambda x: x**2, point)) <= r**2]

    V_approx = (2*r)**d * len(n_d)/len(points)

    return V_approx


#Exc2, real value
def hypersphere_exact(n, d, r=1):
    # n is the number of points
    # d is the number of dimensions of the sphere 

    V_exact = r**d * m.pi**(d/2) / m.gamma(d/2 + 1)   # The formula provided in the instructions

    return V_exact

#Exc3: numba version
#@njit
def sphere_volume_numba(n:int, d:int, r=1)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere

    # Older Numpy .uniform instead of default_rng() since Numba struggled with it
    points = np.random.uniform(-r, r, size=(n, d))  

    n_d = 0   # Start counter of nr of points inside sphere

    # Explicit loops instead of comprehension, since Numba struggled with it
    for point in points:
        distance_squared = 0

        for x in point:
            distance_squared += x**2

        if distance_squared <= r**2:
            n_d += 1

    V_approx = (2*r)**d * n_d/n

    return V_approx

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):

    n_lst = [n // np] * np   # Create a list with np entries of equal size (one for each process)
    d_lst = [d for _ in range(np)]

    with future.ProcessPoolExecutor() as ex:

        results = list(ex.map(sphere_volume, n_lst, d_lst))  # Map sphere_volume to every process

    return sum(results) / len(results)  # Return the mean of all approximations as the final approximation


def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    r = 1

    print(f"Approximate volume of {d} dimensional sphere with radius r = {r} (using n = {n}) = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimensional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    r = 1

    print(f"Approximate volume of {d} dimensional sphere with radius r = {r} (using n = {n}) = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimensional sphere with radius r = {r} = {hypersphere_exact(n,d)}")

    # Exc3: 
    n = 1000000
    d = 11

    times_python = []
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        times_python.append(stop-start)
        print(f"Exc3: Sequential time of d = {d} and n = {n} using conventional Python (call {i+1}): {stop-start}")
    
    times_numba = []
    for i in range(3):
        start_numba = pc()
        sphere_volume_numba(n, d)
        stop_numba = pc()
        times_numba.append(stop_numba-start_numba)
        print(f"Exc3: Sequential time of d = {d} and n = {n} using Numba (call {i+1}): {stop_numba-start_numba}")

    # # Plot the times for Python and Numba together
    # plt.plot([1, 2, 3], times_python, marker="o", label="Conventional Python")
    # plt.plot([1, 2, 3], times_numba, marker="o", label="Numba")

    # plt.xlabel("Run")
    # plt.ylabel("Execution time (s)")
    # plt.title("Execution time comparison")
    # plt.xticks([1, 2, 3])
    # plt.yticks(np.arange(0, 5, 0.5))

    # plt.legend()
    # plt.savefig(f"execution_time_comparison.png")


    # Exc4
    n = 1000000
    d = 11

    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    start_parallel = pc()
    sphere_volume_parallel(n, d)
    stop_parallel = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop_parallel-start_parallel}")
    
    # On local PC
    # Exc4: Sequential time of 11 and 1000000: 4.402295300038531
    # Exc4: Parallel time of 11 and 1000000: 6.063679000129923

    # On Linux Servers
    #Exc4: Sequential time of 11 and 1000000: 14.091906297020614
    #Exc4: Parallel time of 11 and 1000000: 1.5753624450881034

    # Differences
    # Sequential: About 9.7 seconds slower on Linux server
    # Parallel: About 4.5 secound faster on Linux server (approx 3.85 times as fast)
    

if __name__ == '__main__':
	main()
