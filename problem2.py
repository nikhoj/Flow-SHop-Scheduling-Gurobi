import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('crash.txt')

train = data[0::2]
test = data[1::2]
train_x = np.array(train[:, 0]).reshape(len(train), 1)
train_t = np.array(train[:, 1]).reshape(len(train), 1)
test_x = np.array(test[:, 0]).reshape(len(test), 1)
test_t = np.array(test[:, 1]).reshape(len(test), 1)

train_e = np.zeros((5,2))
test_e = np.zeros((5,2))


def RMSE(true, pred):
    
    ew =  ((true - pred) **2)
    ew = 0.5 * np.sum(ew, axis = 0)
    ew = (2 * ew) / len(true)
    erms = np.sqrt(ew)
    
    return erms

def RBF(X,L):
    
    mu, sigma = np.linspace(0,np.max(X), L, retstep=True)
    phi = X - mu
    phi = phi**2
    phi = phi / (2 * sigma ** 2)
    phi = (-1) * phi
    phi = np.exp(phi)
    
    return phi


    

for L in range(5,26,5):
    #training error calculation
    phi = RBF(train_x, L)
    w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
    predict = phi.dot(w)
    idx = int(L / 5)-1
    train_e[idx,0] = L
    train_e[idx,1] = RMSE(train_t, predict)

    #test error calculation
    phi_t = RBF(test_x, L)
    w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
    predict = phi_t.dot(w)
    test_e[idx,0] = L
    test_e[idx,1] = RMSE(test_t, predict)

def plot1():
    plt.figure(figsize = (10,4)) 
    plt.title('RMS')   
    plt.plot(train_e[:,0],train_e[:,1], label='Training')
    plt.plot(test_e[:,0], test_e[:,1], label='Validation')
    #plt.xticks(np.arange(1,21))
    plt.legend()
    plt.show



best_L = test_e[np.argmin(test_e[:,1]),0]

def plot2():
    L = best_L
    phi = RBF(train_x, L)
    w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
    phi_t = RBF(test_x, L)
    predict = phi_t.dot(w)
    plt.figure(figsize = (10,4)) 
    plt.title('Best fit')   
    plt.scatter(train_x, train_t, marker = 'o' , color = 'g' , label = 'Training Data')
    plt.scatter(test_x, test_t, color = 'b', marker = 'o', label = 'validation data')
    plt.plot(test_x, predict, color = 'r', label = 'Validation Best Fit')
    plt.legend()
    plt.show

plot1()
plot2()