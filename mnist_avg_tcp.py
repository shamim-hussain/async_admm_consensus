

import numpy as np
import torch
from admm import Worker, Master
import sys
import threading
import json
from tcp_server import Server, Client
import argparse
import time



class AvgWorker(Worker):
    def __init__(self, X_i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_i = X_i.to(self.device)

        self.x_m = self.X_i.mean(0)
        
    def local_optim(self):
        self.x_i = (self.x_m+self.beta*self.z-self.l_i)/(1+self.beta)


class AvgMaster(Master):
    def __init__(self, X, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X.to(self.device)

    def objective(self, z):
        return 0.5*((self.X-z)**2).mean(0).sum().cpu().item()



def load_data(w_i, num_worker):
    with np.load('mnist.npz') as dat:
        X = dat['X_train']
        
    X = torch.from_numpy(X).float()/255
    
    if w_i == num_worker:
        return X
    else:
        N = X.shape[0]
        N_i = (N+num_worker-1)//num_worker
        
        begin = min(N_i*w_i,N)
        end = min(N_i*(w_i+1),N)
        X_i = X[begin:end]
        return X_i


def run_master(config):
    w_i, num_worker = config.site_id, config.num_worker
    beta, S, tau, steps = config.beta, config.S, config.tau, config.steps
    device = config.device


    X = load_data(w_i, num_worker)
    x_dim = tuple(X.shape[1:])
    
    master = AvgMaster(X, num_worker, x_dim, beta, S, tau, device)
    
    addrport = ('localhost',config.port)
    server = Server(addrport, master.send_queue, master.recv_queue, num_worker)
    server.accept()

    recv_thread=threading.Thread(target=server.recv_loop,daemon=True)
    recv_thread.start()
    
    
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'./logs/mnist_avg/S={S} tau={tau}')
    
    time_vals = []
    z_vals = torch.zeros((steps,)+x_dim, device=device)
    print('Started computing!')
    time_0 = time.time()
    for _ in server.send_iter():    
        if master.stop:
            print('Optimization Ended, calculating objective values.')
            for step in range(steps):
                obj = master.objective(z_vals[step])
                writer.add_scalar('objective', obj, global_step=step+1, walltime=time_vals[step])

            print('Done!')
            
            break
        
        master.receive()
        if master.update():
            z_vals[master.k-1] = master.z
            time_vals.append(time.time()-time_0)
        
        if master.k == steps:
            master.stop_algorithm()
        
        



def run_worker(config):
    w_i, num_worker = config.site_id, config.num_worker
    beta = config.beta
    device = config.device
    
    X = load_data(w_i, num_worker)
    x_dim = tuple(X.shape[1:])
    
    
    worker = AvgWorker(X, w_i, num_worker, x_dim, beta, device)
    
    addrport = ('localhost',config.port)
    client = Client(addrport, worker.send_queue, worker.recv_queue, num_worker,w_i)

    recv_thread=threading.Thread(target=client.recv_loop,daemon=True)
    recv_thread.start()
    
    
    for _ in client.send_iter():    
        if worker.stop:
            break
        
        worker.receive()
        worker.update()
        


# Main function
def main():
    beta = 10.0
    S = 2
    tau = 1
    steps = 50
    device = 'cuda:0'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('num_worker',type=int)
    parser.add_argument('site_id',type=int)
    parser.add_argument('port',type=int)
    parser.add_argument('--beta',type=float,default=beta)
    parser.add_argument('--S',type=int,default=S)
    parser.add_argument('--tau',type=int,default=tau)
    parser.add_argument('--steps',type=int,default=steps)
    parser.add_argument('--device',default=device)
    
    config = parser.parse_args()

    if config.site_id == config.num_worker:
        run_master(config)
    else:
        run_worker(config)




if __name__ == '__main__':
    main()
        
