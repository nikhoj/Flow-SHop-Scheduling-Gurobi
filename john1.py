import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt("crash.txt")
    train, val = data[::2], data[1::2]
    return train, val

train, val = load_data()


def poly_basis_f(x, l):
    phi = np.ones((len(x), l+1))
    for i in range(len(x)):
        phi[i] = np.array([x[i]**j for j in range(l+1)])
    return phi

    
def calc_rms(data, train=train, n=21):
    rms_ = []
    for l in range(1,n):
        phi_t = poly_basis_f(train[:,0], l)
        phi = poly_basis_f(data[:,0], l)
        
        w_hat = np.linalg.solve(phi_t.T.dot(phi_t), phi_t.T.dot(train[:,1]))
        
        t = data[:,1]
        t_hat = phi.dot(w_hat)
        rms = np.sqrt(((t-t_hat)**2).mean())
        rms_.append(rms)
    return rms_, np.argmin(rms_)
    
def plot1(train=train, val=val):
    train, val = calc_rms(train)[0], calc_rms(val)[0]
    plt.xticks(np.arange(1,21))
    plt.plot(train, label="Training")
    plt.plot(val, label="Validation")
    plt.legend()
    plt.title("RMS")
    plt.show()
    


plot1()
