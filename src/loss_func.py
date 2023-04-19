from layer import *
#Loss function 
def reshape_labels(prediction,labels):
    label_reshape = np.zeros(prediction.shape)
    for i in range(0,len(prediction)):
        label_reshape[i][labels[i]] = 1
    return(label_reshape)

def loss_SSE(prediction, labels):
    loss_SSE = []
    labels_reshape = reshape_labels(prediction, labels)
    for i in range(0,len(prediction)):
        loss_SSE.append(np.sum((prediction[i] - labels_reshape[i])**2))
    return(np.mean(loss_SSE))

def grad_loss(prediction, labels):
    label_reshape = reshape_labels(prediction, labels)
    grad_loss_out = []
    for i in range(0,len(prediction)):
        grad_loss_out.append(2*(prediction[i] - label_reshape[i]))
    return(np.array(grad_loss_out))
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def grad_softmax(x):
    '''Compute the gradient of the output layer over each z, Output = softmax(z)
    Input -> Output layer you waht to do the gradient of softmax over
    Outpu -> Matrix of the gradient of each outpu w.r.t each z
    '''
    grad_softmax =[]
    for k in range(0,len(x)):
        jacobian =np.empty((x.shape[-1],x.shape[-1]))
        for i in range(0, jacobian.shape[0]):
            for j in range(0, jacobian.shape[1]):
                if i == j:
                    jacobian[i][j] =  x[k][i] * (1 - x[k][j])
                else:
                    jacobian[i][j] =  x[k][i] * (0 - x[k][j])
        grad_softmax.append(jacobian)            
    return(np.array(grad_softmax))

    #Step 1 - Calculate the Gradient of the loss w.r.t the outputs
    grad_loss_outputs = grad_loss(prediction, labels)

    #Step 2 - Calculate the Gradient of each output w.r.t each z, where output[i] = softmax(z[i])
    grad_outputs_z = grad_softmax(prediction)

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)