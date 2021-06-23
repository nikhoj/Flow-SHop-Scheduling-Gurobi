import numpy as np
import scipy

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
        
    




    

