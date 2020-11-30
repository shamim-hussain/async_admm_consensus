

import numpy as np
import torch
from admm import Worker, Master
import sys, os
import threading
import json
from tcp_server import Server, Client
import argparse
import time

import torch.nn.functional as F


MNIST_SHAPE=28*28
WORKER_LR=1e-2
WORKER_STEPS=100

def W2wb(W):
    return W[:-1], W[-1]

def wb2W(w, b):
    return torch.cat([w, b[None,:]], dim=0)

def log_prob(X, W):
    w, b = W2wb(W)
    return F.log_softmax(X@w + b, dim=-1)

def loss(X, Y, W):
    l = log_prob(X, W)
    return F.nll_loss(l, Y)


class MCWorker(Worker):
    def __init__(self, X_i, Y_i, lr, steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_i = X_i.to(self.device)
        self.Y_i = Y_i.to(self.device)
        self.lr = lr
        self.steps = steps
        
    def local_optim(self):
        try:
            self.x_i.requires_grad = True
            for _ in range(self.steps):
                step_loss = loss(self.X_i, self.Y_i, self.x_i) \
                                + torch.sum(self.l_i*self.x_i) \
                            + 0.5*self.beta*torch.sum((self.x_i-self.z)**2)
                step_loss.backward()
                self.x_i.data -= self.lr * self.x_i.grad
                self.x_i.grad.zero_()
        finally:
            self.x_i.requires_grad = False


class MCMaster(Master):
    def __init__(self, X, Y, X_test, Y_test, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.X = X.to(self.device)
        self.Y = Y.to(self.device)
        self.X_test = X_test.to(self.device)
        self.Y_test = Y_test.to(self.device)

    def objective(self, z):
        w, b = W2wb(self.z)
        train_loss = loss(self.X, self.Y, z)
        test_loss = loss(self.X_test, self.Y_test, z)
        return train_loss, test_loss



def load_data(w_i, num_worker):
    with np.load('mnist.npz') as dat:
        X = dat['X_train']
        Y = dat['Y_train']
        X_test = dat['X_test']
        Y_test = dat['Y_test']
        
    X = torch.from_numpy(X).float()/255
    Y = torch.from_numpy(Y).long()
    X_test = torch.from_numpy(X_test).float()/255
    Y_test = torch.from_numpy(Y_test).long()
    
    X = X[:,::2,::2].reshape(-1,MNIST_SHAPE//4)
    X_test = X_test[:,::2,::2].reshape(-1,MNIST_SHAPE//4)

    xm = X.mean(0)
    xd = X.std(0) + 1e-9
    X = (X - xm)/xd
    X_test = (X_test - xm)/xd

    if w_i == num_worker:
        return X, Y, X_test, Y_test, xm, xd
    else:
        N = X.shape[0]
        N_i = (N+num_worker-1)//num_worker
        
        begin = min(N_i*w_i,N)
        end = min(N_i*(w_i+1),N)
        X_i = X[begin:end]
        Y_i = Y[begin:end]
        return X_i, Y_i


def save_results(save_path, **kwargs):
    vdict = {}
    for k,v in kwargs.items():
        vdict[k] = v.data.cpu().numpy()
    np.savez(save_path, **vdict)



def run_master(config):
    w_i, num_worker = config.site_id, config.num_worker
    beta, S, tau, steps = config.beta, config.S, config.tau, config.steps
    device = config.device


    X, Y, X_test, Y_test, xm, xd = load_data(w_i, num_worker)
    x_dim = (MNIST_SHAPE//4+1,10)
    
    master = MCMaster(X,Y, X_test, Y_test, num_worker, x_dim, beta, S, tau, device)
    
    addrport = ('localhost',config.port)
    server = Server(addrport, master.send_queue, master.recv_queue, num_worker)
    server.accept()

    recv_thread=threading.Thread(target=server.recv_loop,daemon=True)
    recv_thread.start()
    
    # For logging
    import pandas as pd
    log_dir=f'./logs/mnist_logistic/S={S} tau={tau}'
    os.makedirs(log_dir,exist_ok=True)
    
    time_vals = []
    z_vals = torch.zeros((steps,)+x_dim, device=device)
    print('Started computing!')
    time_0 = time.time()
    for _ in server.send_iter():    
        if master.stop:
            print('Optimization Ended, calculating objective values.')
            data = []
            for step in range(steps):
                train_loss, test_loss = master.objective(z_vals[step])

                data.append([step+1, time_vals[step], train_loss.item(), test_loss.item()])
            
            dataframe = pd.DataFrame(data, columns=['global_step', 'wall_time',
                                                    'train_loss', 'test_loss'])
            dataframe.to_pickle(log_dir+'/logs.pkl')
            dataframe.to_csv(log_dir+'/logs.csv')

            w, b = W2wb(master.z)
            save_results(log_dir+'/results.npz', w=w, b=b, xm=xm, xd=xd)                        
            print('Done!')
            break
        
        master.receive()
        if master.update():
            if master.k == 1: time_0 = time.time()
            z_vals[master.k-1] = master.z
            time_vals.append(time.time()-time_0)
        
        if master.k == steps:
            master.stop_algorithm()
        
        



def run_worker(config):
    w_i, num_worker = config.site_id, config.num_worker
    beta = config.beta
    device = config.device
    
    X, Y = load_data(w_i, num_worker)
    x_dim = (MNIST_SHAPE//4+1,10)
    
    
    worker = MCWorker(X, Y, WORKER_LR, WORKER_STEPS, 
                        w_i, num_worker, x_dim, beta, device)
    
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
    beta = 1.0
    S = 2
    tau = 1
    steps = 50
    device = 'cpu:0'
    
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
        
