"""
Assignment 3
Submitted by Md Mahabub Uz Zaman
A20099364
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

def problem1():
    data = np.loadtxt('crash.txt')
    
    train = data[0::2]
    test = data[1::2]
    train_x = np.array(train[:, 0]).reshape(len(train), 1)
    train_t = np.array(train[:, 1]).reshape(len(train), 1)
    test_x = np.array(test[:, 0]).reshape(len(test), 1)
    test_t = np.array(test[:, 1]).reshape(len(test), 1)
    
    train_e = np.zeros((20,1))
    test_e = np.zeros((20,1))
    
    def RMSE(true, pred):
        
        ew =  ((true - pred) **2)
        ew = 0.5 * np.sum(ew, axis = 0)
        ew = (2 * ew) / len(true)
        erms = np.sqrt(ew)
        
        return erms
    
    for L in range(1,21):
        #training error calculation
        phi = train_x**range(L+1)
        w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
        predict = phi.dot(w)
        train_e[L-1] = RMSE(train_t, predict)
    
        #test error calculation
        phi_t = test_x**range(L+1)
        w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
        predict = phi_t.dot(w)
        test_e[L-1] = RMSE(test_t, predict)
    
    def plot1():
        plt.figure(figsize = (10,4)) 
        plt.title('RMS')   
        plt.plot(range(1,21),train_e, label='Training')
        plt.plot(range(1,21), test_e, label='Validation')
        plt.xticks(np.arange(1,21))
        plt.legend()
        plt.show
        
    
    
    
    
    best_L = np.argmin(test_e) + 1
    
    
    def plot2():
        L = best_L
        phi = train_x**range(L+1)
        w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_t))
        phi_t = test_x**range(L+1)
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
    
def problem2():
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
    
def problem3():
    
    
    data = np.loadtxt('crash.txt')
    
    train = data[0::2]
    test = data[1::2]
    train_x = np.array(train[:, 0]).reshape(len(train), 1)
    train_t = np.array(train[:, 1]).reshape(len(train), 1)
    test_x = np.array(test[:, 0]).reshape(len(test), 1)
    test_t = np.array(test[:, 1]).reshape(len(test), 1)
    
    train_e = np.zeros((100,2))
    test_e = np.zeros((100,2))
    
    
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
    
    i = 0
    for alpha in np.logspace(-8,0,100):
        L = 50
        beta = .0025
        #train error calculation
        phi = RBF(train_x, L)
        w = np.linalg.solve(phi.T.dot(phi) + (alpha/beta) * np.identity(50), phi.T.dot(train_t))
        predict = phi.dot(w)
        train_e[i,0] = alpha
        train_e[i,1] = RMSE(train_t, predict)
        
        #test error calculation
        phi_t = RBF(test_x, L)
        predict = phi_t.dot(w)
        test_e[i,0] = alpha
        test_e[i,1] = RMSE(test_t, predict)
        
        i = i+1
        
    best_alpha = test_e[np.argmin(test_e[:,1]),0]
    
    def plot2():
        alpha = best_alpha
        phi = RBF(train_x, 50)
        w = np.linalg.solve(phi.T.dot(phi) + (alpha/beta) * np.identity(50), phi.T.dot(train_t))
        phi_t = RBF(test_x, L)
        predict = phi_t.dot(w)
        plt.figure(figsize = (10,4)) 
        plt.title('Best fit')   
        plt.scatter(train_x, train_t, marker = 'o' , color = 'g' , label = 'Training Data')
        plt.scatter(test_x, test_t, color = 'b', marker = 'o', label = 'validation data')
        plt.plot(test_x, predict, color = 'r', label = 'Validation Best Fit')
        plt.legend()
        plt.show
    
    
    plot2()

def problem4():
    def flower_to_float(s):
        d = {b'Iris-setosa':0.,b'Iris-versicolor':1.,b'Iris-virginica':2.}
        return d[s]
    
    irises = np.loadtxt('iris.data',delimiter=',',converters={4:flower_to_float})
    data = irises[:,0:-1]
    pre_label = irises[:,-1].reshape(150,1)
    label = np.zeros((150,3))
    i = 0
    for value in pre_label:
        value = int(value)
        label[i,value] = 1
        i +=1
    
    temp = np.ones((150,1))
    data = np.hstack((temp, data))
    
    
    
    #train and test set divide
    temp_data = np.hstack((data, label))
    np.random.shuffle(temp_data)
    
    train_data = temp_data[0:75, 0:5] 
    train_label= temp_data[0:75, 5:] 
    test_data= temp_data[75:, 0:5] 
    test_label= temp_data[75:, 5:]
    
    def f(w):
        prior = (0.00313/2) * w.T.dot(w)
        likelihood = 0
        for n in range(train_data.shape[0]):
            
            a = np.sum(train_label[n] * w.reshape(3,5).dot(train_data[n]))
            b = np.sum(np.exp(w.reshape(3,5).dot(train_data[n])))
            
            likelihood += a - np.log(b)
        
        return prior - likelihood
    
    w_init = np.ones(15)
    w_hat = scipy.optimize.minimize(f, w_init).x
    
    
    def predict_accuracy(w, test, test_label):
        
        prediction = np.zeros((test.shape[0],1))
        count = 0
        for n in range(test.shape[0]):
            a = w.reshape(3,5).dot(test[n])
            a = np.exp(a)
            b = w.reshape(3,5).dot(test[n])
            b = np.exp(b)
            b = np.sum(b)
            
            softmax = a/b
            
            prediction[n] = np.argmax(softmax)
            
            count +=1 if test_label[n,int(prediction[n])] == 1 else 0
        
        accuracy = count / len(test) 
            
        return accuracy
    
    prediction = predict_accuracy(w_hat, test_data, test_label)
    
    
    print("Logistic regression accuracy on test set: " + str(prediction))
    
#problem1()
#problem2()
#problem3()
#problem4()