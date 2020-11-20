

import numpy as np
import torch
from admm import Worker, Master
import socket
import sys
import threading
import json
from server import server_loop, send_iter
import argparse
#from types import SimpleNamespace



KNOWNHOSTFILE='knownhosts.json'


class AvgWorker(Worker):
    def __init__(self, X_i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_i = X_i.to(self.device)
        
    def local_optim(self):
        self.x_i = (self.X_i.mean(0)+self.beta*self.z-self.l_i)/(1+self.beta)



def objective(X,z):
    return 0.5*((X-z)**2).mean(0).sum().item()







# Read knownhosts.json
def read_json(site_id, filename):
    try:
        with open(filename,'r') as fp:
            deserialized=json.load(fp)
    except:
        print('Could not open json file!')
        sys.stdout.flush()
        sys.exit()
    
    addrports=[]
    self_proc_id = 0
    hosts_info=deserialized['hosts']
    hosts = {}

    for i, host in enumerate(sorted(hosts_info.keys())):
        hosts[i]=host
        if host==site_id:
            self_proc_id = i
        info = hosts_info[host]
        addrports.append((info['ip_address'], info['udp_start_port']))
    
    return self_proc_id, addrports, hosts



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


def run_master(self_proc_id, addrports, config):
    w_i, num_worker = self_proc_id, len(addrports)-1
    beta, S, tau, steps = config.beta, config.S, config.tau, config.steps
    device = config.device
    
        
    X = load_data(w_i, num_worker)
    x_dim = tuple(X.shape[1:])
    
    master = Master(num_worker, x_dim, beta, S, tau, device)
    
    
    # Start the server loop as a daemon thread
    sever_thread=threading.Thread(target=server_loop,
                                  args=(self_proc_id, addrports,
                                        master.recv_queue), 
                                  daemon=True)
    sever_thread.start()
    
    
    obj_vals = []
    
    for _ in send_iter(addrports, master.send_queue, wait=True):    
        if master.stop:
            print(obj_vals)
            break
        
        master.receive()
        if master.update():
            obj_vals.append(objective(X,master.z))
            if not master.k % 5:
                print(f'Step {master.k} : Objective = {obj_vals[-1]:.5f}')
                sys.stdout.flush()
        
        if master.k == steps:
            master.stop_algorithm()
        
        



def run_worker(self_proc_id, addrports, config):
    w_i, num_worker = self_proc_id, len(addrports)-1
    beta = config.beta
    device = config.device
    
    X = load_data(w_i, num_worker)
    x_dim = tuple(X.shape[1:])
    
    
    worker = AvgWorker(X, w_i, num_worker, x_dim, beta, device)
    
    
    # Start the server loop as a daemon thread
    sever_thread=threading.Thread(target=server_loop,
                                  args=(self_proc_id, addrports,
                                        worker.recv_queue), 
                                  daemon=True)
    sever_thread.start()
    
    
    for _ in send_iter(addrports, worker.send_queue, wait=True):    
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
    parser.add_argument('site_id')
    parser.add_argument('--beta',type=float,default=beta)
    parser.add_argument('--S',type=int,default=S)
    parser.add_argument('--tau',type=int,default=tau)
    parser.add_argument('--steps',type=int,default=steps)
    parser.add_argument('--device',default=device)
    
    config = parser.parse_args()
    
    self_proc_id,addrports,_ = read_json(config.site_id,KNOWNHOSTFILE)

    if self_proc_id == len(addrports)-1:
        run_master(self_proc_id,addrports,config)
    else:
        run_worker(self_proc_id,addrports,config)




if __name__ == '__main__':
    main()
        