# Asynchronous ADMM for Consensus Optimization
>> For example run and more information see the [example_run.ipynb notebook](example_run.ipynb).


## Files Description



*   The main scripts are *mnist_avg.py* and *mnist_logistic.py* which are experiments designed to average the MNIST dataset and run Multiclass Logistic Regression on the MNIST dataset respectively.



>> You need to specify the process id when running these files. 
*   A *(process id) = (number of workers)* corresponds to the master
*   A process id from *0* to *(number of workers - 1)* are different workers

>> The results are generated in the logs directory.


*   The *run_workers.sh* bash script is a tool to spawn multiple processes including all workers and the master.
*   The *plotting_tool.py* file can be used to produce plots from the logs.
*   *mnist.npz* and *mnist_14.npz* are 28x28 and 14x14 versions of the MNIST dataset respectively.
*   *admm.py* and *tcp_server.py* are library files used by the main scripts.
*   *exampl_run.ipynb* demonstrates how tor run and evaluate the experiments.
