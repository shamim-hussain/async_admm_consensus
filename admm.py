
import torch


from collections import deque


class Message:
    def __init__(self, sender, receiver, data):
        self.sender = sender
        self.receiver = receiver
        self.data = data



class Master:
    def __init__(self, num_worker, x_dim, 
                 beta, S, tau,
                 device=None):
        self.num_worker = num_worker
        self.x_dim = x_dim
        self.beta = beta
        self.S = S
        self.tau = tau
        self.device = device
        
        self.send_queue = deque()
        self.recv_queue = deque()  
        
        self.stop = False
        
        self.initialize()
    
    def initialize(self):
        self.u_i = torch.zeros((self.num_worker,),dtype=torch.int32)
        self.t_i = torch.ones((self.num_worker,),dtype=torch.int32)
        
        self.k = 0
        self.x_i = torch.zeros((self.num_worker,)+self.x_dim,
                               device=self.device) 
        self.l_i = torch.zeros((self.num_worker,)+self.x_dim,
                               device=self.device)
        
        self.z = torch.zeros(self.x_dim, device=self.device) 
        
        
    def receive(self):
        for _ in range(len(self.recv_queue)):
            try:
                m = self.recv_queue.popleft()
            except IndexError:
                return
            
            self.u_i[m.sender]=1
            self.t_i[m.sender]=1
            
            self.l_i[m.sender]=m.data[0]
            self.x_i[m.sender]=m.data[1]
    
    
    def update(self):
        if self.u_i.sum()<self.S or self.t_i.max()>self.tau:
            return False
        
        self.z = self.x_i.mean(0)+self.l_i.mean(0)/self.beta
        
        self.send_messages(self.z)
        
        self.t_i = self.t_i + 1 - self.u_i
        self.u_i[:] = 0
        
        self.k+=1
        
        return True
    
    def send_messages(self, data):
        for w_i in range(self.num_worker):
            message = Message(self.num_worker, w_i, data)
            self.send_queue.append(message)
    
    def stop_algorithm(self):
        self.send_messages(None)
        self.stop = True



class Worker:
    def __init__(self, worker_id, num_worker, x_dim, beta, device=None):
        self.worker_id = worker_id
        self.num_worker = num_worker
        self.x_dim = x_dim
        self.beta = beta
        self.device = device
        
        self.send_queue = deque()
        self.recv_queue = deque()
        
        self.stop = False
        
        self.initialize()
        
    
    def initialize(self):
        self.k = 0
        
        self.u_i = 1
        
        self.l_i = torch.zeros(self.x_dim, device=self.device)
        self.x_i = torch.zeros(self.x_dim, device=self.device)
        
        self.z = torch.zeros(self.x_dim, device=self.device) 
    
    
    def receive(self):
        for _ in range(len(self.recv_queue)):
            try:
                m = self.recv_queue.popleft()
            except IndexError:
                return
            
            if m.data is None:
                self.stop = True
            else:
                self.z = m.data
                
                self.u_i = 1
    
    
    def update(self):
        if self.u_i == 0:
            return False
        
        self.l_i = self.l_i + self.beta * (self.x_i - self.z)
        
        self.local_optim()
        
        data = torch.stack([self.l_i, self.x_i])
        self.send_messages(data)
        
        self.u_i = 0
        
        self.k += 1
        
        return True
    
    
    def send_messages(self,data):
        message = Message(self.worker_id, self.num_worker, data)
        self.send_queue.append(message)
    
    
    def local_optim(self):
        pass











        