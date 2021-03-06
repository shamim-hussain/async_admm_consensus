# Asynchronous ADMM for Consensus Optimization
An implementation of the Asynchronous ADMM algorithm for Consensus Optimization presented in the paper 

[*Zhang, Ruiliang, and James Kwok. "Asynchronous distributed ADMM for consensus optimization." International conference on machine learning. 2014.*](http://proceedings.mlr.press/v32/zhange14.pdf)

in Python using PyTorch and TCP sockets.

## Presentation and Problem Set

* The recorded presentation can be found as [*presentation.mp4*](presentation.mp4).
* The slide deck is contained in the [*slides.pdf*](slides.pdf).
* The problem set can be found in [*Problem_Set.pdf*](Problem_Set.pdf).
* The solutions are given in [*Problem_Set__With_Solutions.pdf*](Problem_Set__With_Solutions_.pdf).

## Implementation
>> For example run and more information see the [example_run.ipynb notebook](example_run.ipynb).


### Files Description



*   The main scripts are *mnist_avg.py* and *mnist_logistic.py* which are experiments designed to average the MNIST dataset and run Multiclass Logistic Regression on the MNIST dataset respectively.



>> You need to specify the process id when running these files. 
>> *   A *(process id) = (number of workers)* corresponds to the master
>> *   A process id from *0* to *(number of workers - 1)* are different workers
>> The results are generated in the logs directory.


*   The *run_workers.sh* bash script is a tool to spawn multiple processes including all workers and the master.
*   The *plotting_tool.py* file can be used to produce plots from the logs.
*   *mnist.npz* and *mnist_14.npz* are 28x28 and 14x14 versions of the MNIST dataset respectively.
*   *admm.py* and *tcp_server.py* are library files used by the main scripts.
*   *exampl_run.ipynb* demonstrates how tor run and evaluate the experiments.
