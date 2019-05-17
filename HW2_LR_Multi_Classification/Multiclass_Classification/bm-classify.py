import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2
    
    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
     
    

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        X = np.insert(X,0,[1],axis = 1)
        np.place(y, y == 0, [-1])
        
        sign = np.zeros(N)
        
        
        for i in range(max_iterations):
            
        
            w = np.insert(w,0,[b])
            
            m_1 = np.matmul(X,w)
            m_2 = np.multiply(y,m_1)
            m_3 = m_2<=0
            sign = m_3.astype(int)
            
            '''
            for j in range(N):
                test = m_2[j] <= 0
                if test == True:
                    test = 1
                else:
                    test = 0
                sign[j] = np.sign(test)
            
            for j in range(N):
                m_1 = np.matmul(X[j],w)
                m_2 = y[j] * m_1
                test = m_2 <= 0
                if test == True:
                    test = 1
                else:
                    test = 0
                sign[j] = np.sign(test)
            '''
            
            a = sign * step_size
            b = np.multiply(a,y)
            c = np.matmul(np.transpose(b),X)
            #print(c)
            
            
                
            w = w + c/N
            
                
            b = w[0]
            w = w[1:]
            
        
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        
        ############################################
        
        X = np.insert(X,0,[1],axis = 1)
        np.place(y, y == 0, [-1])
        
        for i in range(max_iterations):
        
            w = np.insert(w,0,[b])
                
            m_1 = np.matmul(X,w)
            m_2 = np.multiply(y,m_1)
            
            prob = sigmoid(-m_2)
            
            a = prob * step_size
            b = np.multiply(a,y)
            c = np.matmul(np.transpose(b),X)
            #print(c)
            
            
                
            w = w + c/N
            
                
            b = w[0]
            w = w[1:]
                
        
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1.0/(1.0 + np.exp(-1.0 * z))
    
    
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        m_1 = np.matmul(X, w) + b
        
        np.place(m_1, m_1 > 0, [1])
        np.place(m_1, m_1 <=0, [0])
        preds = m_1
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        
        m_1 = sigmoid(np.matmul(X, w) + b)
        np.place(m_1, m_1 > 0.5, [1])
        np.place(m_1, m_1 <= 0.5, [0])
        preds = m_1
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        
        #targets = y.reshape(-1)
        #y_new = np.eye(C)[targets]
        
        X = np.insert(X,D,[1],axis = 1)
        
        #w_new = np.zeros((C,D+1))
        
        
            
        for i in range(max_iterations):
            
            n = np.random.choice(range(N))
            
            w = np.insert(w,D,b,axis = 1)
            #print(X[n])
            #print(w)
            
            pro = np.matmul(w, np.transpose(X[n]))
            #print(pro)
            
            #print(pro)
            
            #pro = pro - np.max(pro)
            
            pro_1 = softMax(pro)
            
            pro_1 = pro_1[:,np.newaxis]
            
            XX = X[n]
            XX = XX[:,np.newaxis]
            
            
            #print(pro_1)
            
            pro_1[y[n]] = pro_1[y[n]] - 1
            
            pro_2 = np.matmul(pro_1, np.transpose(XX))
            
            #print(pro_2)
            
            #pro_2[y[n]] = pro_2[y[n]] - 1
            
            #print(pro_2)
            
            w = w - step_size * pro_2
            
            #print(w)
            
            b = w[:, D]
            w = w[:, :-1]
            
           

          
            
            
            '''w_new = np.zeros((C,D+1))
            
            
            w = np.insert(w,D,b,axis = 1)
            
            
            for k in range(C):
                if k!=y[n]:
                    w_1 = w[k] - w[y[n]]
                    
                    w_2 = np.multiply(X[n],w_1)
                    w_2 = w_2 - np.max(w_2)
                    w_3 = np.exp(w_2)
                    w_44 = np.zeros(D+1)
                    for k1 in range(C):
                        if k1!=y[n] :
                           w_11 = w[k1] - w[y[n]]
                           w_22 = np.multiply(X[n],w_11)
                           w_22 = w_22 - np.max(w_22)
                           w_33 = np.exp(w_22) 
                           w_44 = w_44 + w_33
                    w_44 = w_44 + 1
                    
                    w_5 = np.divide(w_3, w_44)
                    w_6 = np.multiply(w_5,X[n])
                    w_new[k] = w_6
                else:
                    w_44 = np.zeros(D+1)
                    for k1 in range(C):
                        if k1!=y[n] :
                           w_11 = w[k1] - w[y[n]]
                           w_22 = np.multiply(X[n],w_11)
                           w_22 = w_22 - np.max(w_22)
                           w_33 = np.exp(w_22) 
                           w_44 = w_44 + w_33
                    w_55 = -w_44
                    w_44 = w_44 + 1
                    w_66 = np.divide(w_55, w_44)
                    w_6 = np.multiply(w_66,X[n])
                    w_new[k] = w_6'''
                    
                    
            
            
                     
                    
    elif gd_type == "gd":
        
        X = np.insert(X,D,[1],axis = 1)
        
        #w_new = np.zeros((C,D+1))
        targets = y.reshape(-1)
        y_new = np.eye(C)[targets]
        
            
        for i in range(max_iterations):
            
            
            w = np.insert(w,D,b,axis = 1)
            #print(X[n])
            #print(w)
            
            pro = np.matmul(X, np.transpose(w))
            #print(pro)
            
            '''for j in range(N):
                pro[j, :] = pro[j, :] - np.max(pro[j, :])'''
                
            pro = pro - pro.mean(axis=1, keepdims=True)
                
            #print(pro)
            
            pro_1 = softMax_1(pro, N)
            #print(pro_1)
            
            '''for z in range(N):
                pro_1[z][np.where(y_new[z] == 1)] = pro_1[z][np.where(y_new[z] == 1)] - 1
                
            pro_1[:, np.where(y_new[:, ] == 1)]'''
            
            
            pro_1[np.where(y_new[:, ] == 1)] = pro_1[np.where(y_new[:, ] == 1)] - 1
            
            pro_2 = np.matmul(np.transpose(pro_1), X)
            
            w = w - step_size * pro_2/N
            
            b = w[:, D]
            w = w[:, :-1]
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def softMax (X):
    e_x = np.exp(X - np.max(X))
    return e_x / np.sum(e_x, axis = 0)

def softMax_1 (X, N):
    #print(X)
    #e_x = np.exp(X)
    #print(e_x)
    X = np.exp(X)
    
    X = X / X.sum(axis = 1, keepdims=True)
        
    return X


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    X = np.insert(X,0,[1],axis = 1)
    w = np.insert(w,0,b,axis = 1)
    m_1 = np.matmul(X, np.transpose(w))
    #m_1 = np.array([[21,1,1,3],[22,2,5,1],[22,23,24,25]])
    preds = np.argmax(m_1, axis = 1) 

    assert preds.shape == (N,)
    return preds




        