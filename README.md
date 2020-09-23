# MLP_SOLVER
Sequential and parallel version of deterministic local search heuristic is coded to solve MLP.

Source codes are categorized in three files, sequential DLSH (DLSH.c), parallel DLSH (PDLSH.cu), and parallel DLSH using floating-point calculation (PDLSH_v1.cu). For DLSH.c and PDLSH.cu, solution latency is calculated using integers. PDLSH.cu is implemented using both reduction methods, namely built-in and vector. Also, it contains three vector types, namely single-thread, one-pass, and two-pass. For PDLSH_v1.cu, one-pass vector reduction is used.  

For parallel:
1. Compilation command: nvcc PDLSH.cu -arch=sm_35 -o pdlsh
2. Execution command: ./pdlsh ../TSPLIB/TSPLIB/kroA200.tsp

prerequisite:
1. In CUDA code, a 64-bit atomicMin() function has been used that supports on a GPU device which has computing capability 3.5 and higher. 

For sequential:
1. Compilation comand: gcc DLSH.c -lm -o dlsh
2. Execution command: ./dlsh ../TSPLIB/TSPLIB/kroA200.tsp 

TRP instannces are available at following link:
http://antor.uantwerpen.be/instances-in-the-paper-efficient-grasp-vnd-and-grasp-vns-metaheuristics-for-the-traveling-repairman-problem/

If you find this helpful and writing a research paper, you can encourage us citing this work as
"Pramod Yelmewad and Basavaraj Talawar, Parallel deterministic local search heuristic for minimum latency problem, cluster computing, DOI: https://doi.org/10.1007/s10586-020-03173-4"
