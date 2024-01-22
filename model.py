## Jittor


import logging
import jittor as jt
from jittor import nn
from jittor import Module
from jittor import optim
import pygmtools as pygm
from pygmtools.linear_solvers import sinkhorn
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import os
import itertools

from split import *

PERMUTATIONS = list(itertools.permutations(range(4),4))
PERM_ARRAY = np.array(PERMUTATIONS) #[24,4]
PERM_DICT = {value: index for index, value in enumerate(PERMUTATIONS)  }
## Define Network

class LargeCNN(nn.Module):
    def __init__(self, config):
        super(LargeCNN, self).__init__()
        
        self.num_classes = 512    ## 512
        self.in_channels = config["n_input"][0]  ## 3
        # self.in_size = config["n_input"][1]      ## 16

        self.features = nn.Sequential(                                      # 3x16x16
            nn.Conv(self.in_channels, 64, kernel_size=9, stride=1, padding=3,dilation=0),   # 64x14x14
            nn.Relu(), 
            nn.Pool(kernel_size=3, stride=1, op='maximum'),                 
            nn.Conv(64, 192, kernel_size=5, padding=2),                     
            nn.Relu(), nn.Pool(kernel_size=3, stride=1, op='maximum'),      
            nn.Conv(192, 384, kernel_size=3, padding=1), 
            nn.Relu(), 
            nn.Conv(384, 256, kernel_size=3, padding=1), 
            nn.Relu(), 
            nn.Conv(256, 256, kernel_size=3, padding=1),                    
            nn.Relu(), 
            nn.Pool(kernel_size=3, stride=1, op='maximum')  
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))         # No need to calculate precise size 
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(((256 * 6) * 6), 4096), 
            nn.Relu(), 
            nn.Dropout(), 
            nn.Linear(4096, 4096), 
            nn.Relu(), 
            nn.Linear(4096, self.num_classes)
        )
        self.tail = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4,4096),
            nn.Relu(),
            nn.Dropout(),
            nn.Linear(4096,16),
        )

    def execute(self, x):
        # (N, 4, 3, 16, 16)
        # batch, patch, channel, row, col
        N = x.shape[0]
        x = x.reshape(4*N,self.in_channels,16,16)

        x = self.features(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], (- 1)))      # ret: (4*N,256*6*6)
        x = self.classifier(x)                      # ret: (4*N,512)

        x = x.reshape(N,4*512)                      # ret: (N,4*512)                  
        x = self.tail(x)                            # ret: (N,16)
        return x


class SmallCNN(nn.Module):
    def __init__(self, config):
        super(SmallCNN, self).__init__()
        
        self.num_classes = 512    ## 512
        self.in_channels = config["n_input"][0]  ## 3
        # self.in_size = config["n_input"][1]      ## 16

        self.features = nn.Sequential(                                      # 3x16x16
            nn.Conv(self.in_channels, 64, kernel_size=9, stride=1, padding=3,dilation=0),   # 64x14x14
            nn.Relu(), 
            nn.Pool(kernel_size=3, stride=1, op='maximum'),                 
            nn.Conv(64, 192, kernel_size=5, padding=2),                     
            nn.Relu(), nn.Pool(kernel_size=3, stride=1, op='maximum'),      
            nn.Conv(192, 256, kernel_size=3, padding=1), 
            nn.Relu(), 
            nn.Pool(kernel_size=3, stride=1, op='maximum')  
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))         # No need to calculate precise size 
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(((256 * 6) * 6), 4096), 
            nn.Relu(), 
            nn.Dropout(), 
            nn.Linear(4096, 2048), 
            nn.Relu(), 
            nn.Linear(2048, self.num_classes)
        )
        self.tail = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4,4096),
            nn.Relu(),
            nn.Dropout(),
            nn.Linear(4096,16),
        )

    def execute(self, x):
        # (N, 4, 3, 16, 16)
        # batch, patch, channel, row, col
        N = x.shape[0]
        x = x.reshape(4*N,self.in_channels,16,16)

        x = self.features(x)
        x = self.avgpool(x)
        x = jt.reshape(x, (x.shape[0], (- 1)))      # ret: (4*N,256*6*6)
        x = self.classifier(x)                      # ret: (4*N,512)

        x = x.reshape(N,4*512)                      # ret: (N,4*512)                  
        x = self.tail(x)                            # ret: (N,16)
        return x


class RealSmallCNN(Module):
    def __init__(self, config):
        super(RealSmallCNN, self).__init__()
        self.config = config
        self.in_channels = config["n_input"][0]  ## 3
        self.dropout = config["dropout"]

        self.features = nn.Sequential(                                      # 3x16x16
            nn.Conv(self.in_channels, 32, kernel_size=3, padding=1),   # 64x14x14  
                nn.Relu(), nn.Dropout(self.dropout),            
            nn.Pool(kernel_size=2, stride=2, op='maximum'),   
            nn.Conv(32, 64, kernel_size=3, padding=1),nn.Relu(),nn.Dropout(self.dropout),
            nn.Conv(64, 128, kernel_size=3, padding=1), nn.Relu(), nn.Dropout(self.dropout), 
            nn.Pool(2,2, op='maximum'),                  
            nn.Conv(128, 256, kernel_size=3, padding=1), nn.Relu(),nn.Dropout(self.dropout), 
            nn.Conv(256, 256, kernel_size=3, padding=1),  nn.Relu(),nn.Dropout(self.dropout),     
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(((2 * 2) * 256), 2048), nn.Relu(), nn.Dropout(self.dropout), 
            nn.Linear(2048, 1024), 
        )
        self.tail = nn.Sequential(
            nn.Linear(1024*4,8192),nn.Relu(),nn.Dropout(self.dropout),
            nn.Linear(8192,4096),nn.Relu(),nn.Dropout(self.dropout),
            nn.Linear(4096,16),#nn.Relu()
            
        )
    def execute(self, x):
        N = x.shape[0]
        x = x.reshape(4*N,self.in_channels,16,16)

        x = self.features(x)
        # x = self.pool2(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        x = x.reshape(N,4*1024)                                     
        x = self.tail(x)                            # ret: (N,16)
        return x

class LTHAlexNet(nn.Module):
    def __init__(self):
        super(LTHAlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv(3, 32, 3, padding=1), nn.Pool(2, stride=2, op='maximum'), nn.ReLU(),nn.Dropout(p=0.1),
            nn.Conv(32, 64, 3, padding=1), nn.Pool(2, stride=2, op='maximum'), nn.ReLU(),nn.Dropout(p=0.1),
            nn.Conv(64, 128, 3, padding=1),nn.Dropout(p=0.2),
            nn.Conv(128, 256, 3, padding=1),nn.Dropout(p=0.2),
            nn.Conv(256, 256, 3, padding=1), nn.Pool(2, stride=2, op='maximum'), nn.ReLU(),nn.Dropout(p=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(((256 * 2) * 2), 2048),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(2048, 1024),nn.ReLU(),nn.Dropout(p=0.1),
        )
    def execute(self, x):
        x = self.feature_extraction(x)
        x = x.view(((- 1), ((256 * 2) * 2)))
        x = self.classifier(x)
        return x
    
class LTHDeepPermNet(nn.Module):
    def __init__(self):
        super(LTHDeepPermNet, self).__init__()
        self.alexnet=LTHAlexNet()
        self.fc = nn.Sequential(
            nn.Linear((1024*4),8192),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(8192,4096),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(4096,16)
        )
       
    def execute(self, x):
        # [N, 4, 3 ,16,16 ]
        # print(x.dtype)
        # x = x.transpose(0,1,3,4,2)
        x1,x2,x3,x4=x[:,0],x[:,1],x[:,2],x[:,3]
    # def execute(self, x1,x2,x3,x4):
        x1,x2,x3,x4=self.alexnet(x1),self.alexnet(x2),self.alexnet(x3),self.alexnet(x4)
        x=jt.concat([x1,x2,x3,x4],dim=1)
        x=self.fc(x)
        x=x.view(-1,16)
        # x=x.view(-1,4,4)
        return x

## Define Algo

def my_cel(output, target):
    # input: (N, 4), [0,1]
    # target: (N, )
    target = (target.reshape(-1)).broadcast(output, [1])
    target = target.index(1) == target
    loss = (0 - ((output+1e-6).log()*target).sum(1))
    return loss.mean()

def my_mse(output, target):
    # input: (N, 4), [0,1]
    # target: (N, 1)
    target=get_permutation_matrix(target)
    loss = (output-target).pow(2).sum(1)
    return loss.mean()

def relate(input):
    # input: [N,4,4]
    perm_array = jt.Var(PERMUTATIONS)   # [24,4]
    perm_matrix = get_permutation_matrix(perm_array)    # [24,4,4]
    output = input.reshape(-1,16).matmul( perm_matrix.reshape(-1,16).transpose(1,0) ) #  [N,16]*[16,24]

    return output

class DPN_Algo():
    def __init__(self, config, model, train_dataloader, dev_dataloader=None, test_dataloader=None):
        self.model = model
        self.config = config
        self.lr = config["lr"]
        self.betas = config["betas"]
        self.batch_size = config["batch_size"]

            ## Assume dataloader is a iter of (N, batch, 4,3,16,16), as original data 
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        if config["criterion"] == "jittor_cel":
            self.criterion = nn.CrossEntropyLoss()
        elif config["criterion"] == "my_cel":
            self.criterion = my_cel
        elif config["criterion"] == "my_mse":
            self.criterion = my_mse
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas=self.betas)
        if self.config["enable_scheduler"]:
            self.scheduler = jt.lr_scheduler.MultiStepLR(self.optimizer, milestones=config["milestones"], gamma=config["gamma"])


        self.train_loss_list = []
        self.train_acc_list = []
        self.dev_loss_list = []
        self.dev_acc_list = []

        self.best_acc = 0
        self.best_epoch = 0
        
    def save(self, path):
        info = f"Model Save at {path}.model/.optim, "
        print(info)
        logging.info(info)
        
        # jt.save(self.model.state_dict(), path+".model")
        self.model.save(path+".model")
        jt.save(self.optimizer.state_dict(), path+".optim")

    def load(self, path):
        # self.model.load_state_dict(jt.load(path))
        self.model.load(path+".model")
        self.optimizer.load_state_dict(jt.load(path+".optim"))

    def train_epoch(self,epoch):  
        self.model.train()
        loss_list = []
        acc_list = []  #DEBUG:
        
        for batch_idx, (data) in enumerate(self.train_dataloader):
            # data  : [N,4,3,16,16]
            data = data[0]
            X_tilde, label = shuffle_permutation(data,self.config)
            output = self.model(X_tilde).reshape(-1,4,4)   # [N,4,4]

            if self.config["method"]=="classic":
                if self.config["enable_sinkhorn"]:
                    output = sinkhorn(output)              
                loss = self.criterion(output.reshape(-1,4),jt.Var(label).reshape(-1,1))
                predict = self._inference(output)          ## predict ,label are numpy array, int
                acc = (np.all(predict == label,axis=1)).sum().item() / len(label)
            elif self.config["method"]=="relate":
                output = relate(output) #[N,24]
                loss = self.criterion(output,jt.Var(label).reshape(-1,1))
                predict = self._inference(output,relate_return_ind=True)          ## predict ,label are numpy array, int
                acc = (predict == label).sum().item() / len(label)

                pass
            self.optimizer.step(loss)
            self.optimizer.zero_grad()

            loss_list.append(loss.item())
            acc_list.append(acc)

        if self.config["enable_scheduler"]:
            self.scheduler.step()

        return np.mean(loss_list), np.mean(acc_list)
        pass

    def train(self, start_epoch=0):
        start_time = time.time()
        self.train_loss_list = []
        self.train_acc_list = []
        self.dev_loss_list = []
        self.dev_acc_list = []

        if start_epoch > 0:
            self.train_loss_list += [0]*start_epoch
            self.train_acc_list += [0]*start_epoch
            self.dev_loss_list += [0]*(start_epoch//self.config["eval_epoch"])
            self.dev_acc_list += [0]*(start_epoch//self.config["eval_epoch"])

        output_info = f"Start Training: {time.asctime()}"
        print(output_info)
        logging.info(output_info)

        for epoch in range(start_epoch,self.config["epoch"]):
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            
            if self.dev_dataloader is not None and ((epoch) % self.config["eval_epoch"] == 0 or epoch == self.config["epoch"] - 1):
                dev_loss, dev_acc = self.evaluate(self.dev_dataloader)
                self.dev_loss_list.append(dev_loss)
                self.dev_acc_list.append(dev_acc)

                if dev_acc > self.best_acc:
                    self.best_acc = dev_acc
                    self.best_epoch = epoch
                    self.save(os.path.join(self.config["save_dir"],f"{int(start_time)%int(1e6)}_epoch{epoch}") ) 
            
            self.plot(epoch)
            epoch_time = time.time()
            output_info = f"Epoch {epoch}: train_loss {round(train_loss, 6)}, train_acc {round(train_acc, 3)}, dev_loss {round(dev_loss, 6)}, dev_acc {round(dev_acc, 3)}, \n\t time {round((epoch_time - start_time)/(epoch+1), 3)} s/epoch, total time {round(epoch_time - start_time, 3)} s, time left {round((epoch_time - start_time)/(epoch+1)*(self.config['epoch']-epoch-1), 3)} s"
            print(output_info)
            logging.info(output_info)

        epoch_time = time.time()
        output_info = f"Best epoch: {self.best_epoch}, best acc: {self.best_acc}\n"+\
    f"Final train_loss: {round(self.train_loss_list[-1],6)} dev_loss: {round(self.dev_loss_list[-1],6)}\n"+\
    f"Final train_acc: {round(self.train_acc_list[-1],6)} dev_acc: {round(self.dev_acc_list[-1],6) }\n" +\
    f"Config: {self.config}\n"
        
        print(output_info)

        logging.info(f"Model: {self.model.modules()[0]} , Params num: {sum([p.numel() for p in self.model.parameters()])}")
        logging.info(output_info)

    def continue_train(self, last_epoch, path):       
        self.load(path=path)
        self.scheduler.last_epoch = last_epoch
        self.train(start_epoch=last_epoch)


    def test(self):
        loss, acc = self.evaluate(self.test_dataloader)
        info = f"Test: loss {round(loss, 6)}, acc {round(acc, 6)}"
        print(info)
        logging.info(info)

    def evaluate(self,dataloader): 
        ## dataloader: [N,4,3,16,16]
        ## Return: (loss, acc)
        self.model.eval()
        loss_list = []
        acc_list = [] #DEBUG:
        for batch_idx, (data) in enumerate(dataloader):
            data = data[0]
            X_tilde, label = shuffle_permutation(data,self.config)
            output = self.model(X_tilde).reshape(-1,4,4)   # [N,4,4]
            if self.config["method"]=="classic":
                if self.config["enable_sinkhorn"]:
                    output = sinkhorn(output)              
                loss = self.criterion(output.reshape(-1,4),jt.Var(label).reshape(-1,1))
                predict = self._inference(output)          ## predict ,label are numpy array, int
                acc = (np.all(predict == label,axis=1)).sum().item() / len(label)
            elif self.config["method"]=="relate":
                output = relate(output) #[N,24]
                loss = self.criterion(output,jt.Var(label).reshape(-1,1))
                predict = self._inference(output,relate_return_ind=True)          ## predict ,label are numpy array, int
                acc = (predict == label).sum().item() / len(label)


            loss_list.append(loss.item())
            acc_list.append(acc)
        return np.mean(loss_list), np.mean(acc_list)
        pass

    def _inference(self,Q,type="greedy",relate_return_ind=False):
        ## return: [N,4]
        if self.config["method"]=="classic":
            ## Q: [N,4,4]
            if type == "greedy":
                ress = np.argmax(Q.numpy(),axis=2)
            elif type == "Frobenius":
                ress = []
                for i in range(Q.shape[0]):
                    P = cp.Variable(shape=(4,4),boolean=True)
                    objective = cp.Minimize(cp.norm(Q[i].numpy()-P, p='fro'))
                    constraints = [cp.sum(P, axis=0) == np.ones(4), cp.sum(P, axis=1) == np.ones(4)]
                    prob = cp.Problem(objective, constraints)
                    prob.solve()
                    res = np.round(P.value).astype(np.int32())
                    ress.append(res.reshape(1,4,4))
                return np.argmax(np.concatenate(ress,axis=0),axis=2)
                # TODO: 
        elif self.config["method"]=="relate":
            ## Q:[N,24]
            ress = np.argmax(Q.numpy(),axis=1) #[N,],np
            if not relate_return_ind:
                ress = PERM_ARRAY[ress]

        return ress


    def inference(self,X_tilde, type="greedy",relate_return_ind=False): 
        ## X_tilde: [N,4,3,16,16]
        ## return: [N,4]
        output = self.model(X_tilde).reshape(-1,4,4)   # [N,4,4]
        
        if self.config["method"]=="classic":
            if self.config["enable_sinkhorn"]:
                output = sinkhorn(output)
        elif self.config["method"]=="relate":
            output = relate(output) #[N,24]

        predict = self._inference(output,type=type,relate_return_ind=relate_return_ind)
        return predict

    def plot(self, epoch=None):
        clear_output(True)
        jt.sync_all(True)

        if epoch is None:
            epoch = self.config['epoch']-1

        fig = plt.figure(figsize=(8, 3.5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.set_ylim(0, 1)
        ax1.set_xlim(0, self.config['epoch'])
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        
        ax1.plot(list(np.arange(0, len(self.train_loss_list))),
                self.train_loss_list, label="train_loss",color = "red",linestyle=":")
        ax2.plot(list(np.arange(0, len(self.train_acc_list))),
                self.train_acc_list, label="train_acc",color = "brown",linestyle="--")
        
        if self.dev_dataloader is not None:
            ## plot
            ax1.plot(list(self.config['eval_epoch'] * np.arange(0, len(self.dev_loss_list))),
                    self.dev_loss_list, label="dev_loss",color="blue",linestyle=":")
            ax2.plot(list(self.config['eval_epoch'] * np.arange(0, len(self.dev_acc_list))),
                    self.dev_acc_list, label="dev_acc",color = "darkblue",linestyle="--")
 

        fig.legend(loc='upper right')
        plt.show()
    

def get_model(config):
    if config['model'] == 'smallcnn':
        model=SmallCNN(config)
    elif config['model'] == "realsmallcnn":
        model = RealSmallCNN(config)
    elif config['model'] == "lth":
        model = LTHDeepPermNet()
        pass

    return model
