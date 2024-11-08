import numpy as np
from random import random

""" 

dE/DWi =(y - y[i+1]) S'(x[i+1])xi
S' (x[i+1])=S(x[i+1])(1-s(x[i+1)))
s(x[i+1]=x[i+1]
x[i+1]=yiWi

"""


#because a lot of data will be needed, we will use a class approach.

class Full_NN(object):
    #A Multi Layer Neural Network class. We use this as for the way we need to handle the
    #variables is better suited.
    def __init__(self, X=2, HL=[2,2], Y=2): #a constructor for some default values.
        self.X=X #inputs
        self.HL=HL #hidden layers
        self.Y=Y #outputs

        #we are setting up some class variables for our inputs.
        L=[X]+HL+[Y] #total number of layers. This creates a representation of the
        #the network in the format we need it. i.e array of the format [how many inputs, how mnay hidden layers. how many outputs]
        W=[] #initialize a weight array


        for i in range(len(L)-1): #we want to be able go to the next layer up so we set one minus
            w=np.random.rand(L[i], L[i+1]) #fill them up with random values, that is why we need the numpy library
            W.append(w) #add the new values to the array.
        self.W=W #link the class variable to the current variable
        
        #initialize a derivative array. This are needed to calculate the 
        # back propagation. they are the derivatives of the activation function
        Der=[] 
        for i in range(len(L) - 1):
            #same reason as above for every line
            #we don't need random values, just to have them ready to be used. we fill up with zeros
            d = np.zeros((L[i], L[i+1])) 
            Der.append(d)
        self.Der = Der

        #we will be passing these here as that way the class variable will keep them for us until we need them.
        
        out = [] # Output array
        for i in range(len(L)): # #We don't need to go +1. The outputs are straight forward.
            o = np.zeros(L[i]) #we don't need random values, just to have them ready to be used. we fill up with zeros.
            out.append(o)
        self.out = out

    def FF(self, x): # method will run the network forward
        out = x # input layer output is just the input
        
        self.out[0] = x # Linking the outputs to the class variable for back propagation. Begin with input layer
        
        for i, w in enumerate(self.W): # go through the network layer via the weights variable
            Xnext = np.dot(out,w) # product between weights and output for the next output
            out = self.sigmoid(Xnext) # use the activation function
            self.out[i+1] = out # pass result to the class variable to preserve for later. Back propagation
        
        return out
    
    def BP(self, Er): # Back propagation method. using the output error *Er* to go backwards through the layers and calculate...
        # the errors needed to update the weights
        # This will return the final error of the input
        
        for i in reversed (range(len(self.Der))): # Go backwards *reversed* through the network
            # In this loop we are going backwards through the layers based on the following equations
            """ 

            dE/DWi =(y - y[i+1]) S'(x[i+1])xi
            S' (x[i+1])=S(x[i+1])(1-s(x[i+1)))
            s(x[i+1]=x[i+1]
            x[i+1]=yiWi

            """
            
            out = self.out[i+1] # Get the layer output for the previous layer (we go reverse)
            
            D = Er * self.sigmoid_Der(out) # Applying the derivative of the activation function to get delta.delta
            # This is essentially - dE/DWi =(y - y[i+1]) S'(x[i+1])xi
            
            D_fixed = D.reshape(D.shape[0], -1).T # Turn Delta into an array of appropriate size
            
            this_out = self.out[i] # current layer output
            
            this_out = this_out.reshape(this_out.shape[0], -1) # reshape as before to get column array suitable for the
            # multiplication we need
            
            self.Der[i] = np.dot(this_out, D_fixed) # Matrix multiplication and pass result to class variable
            
            Er = np.dot(D, self.W[i].T) # This back propagates the next error we need for the next iteration
            # this error term is part of the dE/DWi equation for the next layer down in the back propagation
            # and we pass it on after calculating it in this iteration
            
    def train_nn(self, x, target, epochs, lr): # training the network, x = array, target = array, epochs = int, lr = int
        for i in range(epochs): # training loop for as many epochs as we need
            S_errors = 0 # variable to carry the error we need to report to the user
            
            for j, input in enumerate(x): #iterate through the training data and inputs
                t = target[j]
                output = self.FF(input) # use the network calculations for forward calculations
                
                e = t - output # obtain the overall Network output error
                self.BP(e) # use error to do the back propagation
                self.GD(lr) #Do gradient descent
                
                S_errors += self.msqe(t,output) # updates the overall error to show the user
    
    def GD(self, lr=0.05): #Gradient descent
        for i in range(len(self.W)): # go through the weights
            W = self.W[i]
            Der = self.Der[i]
            W += Der * lr # update the weights by applying the learning rate
            
    def sigmoid(self, x): # sigmoid activation function
        y = 1.0/(1 + np.exp(-x))
        return y
    
    def sigmoid_Der(self, x): # sigmoid function derivative
        sig_der = x * (1.0 - x)
        return sig_der
    
    def msqe(self, t, output): # mean square error
        msq = np.average((t - output) ** 2)
        return msq
    
if __name__ == "__main__": # Testing class
    
    # Essentially teaching a neural network multiplication
    training_inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)]) # Creates a training set of inputs
    targets = np.array([[i[0] * i[1]] for i in training_inputs]) # Creates a training set of outputs
    
    nn = Full_NN(2, [5, 5], 1) # Creates a NN with 2 inputs, 2 hidden layers and 1 output
    
    nn.train_nn(training_inputs, targets, 50, 0.1) # Train network with 0.1 learning rate for 10 epochs
    
    input = np.array([0.3, 0.2]) # After training this tests the train network
    
    target = np.array([0.06]) # Target value
    
    NN_output = nn.FF(input)
    
    print("=============== Testing the Network Screen Output===============")
    print ("Test input is ", input)
    print()
    print("Target output is ",target)
    print()
    print("Neural Network actual output is ",NN_output, "there is an error (not MSQE) of ",target-NN_output, "Actual = ", 0.3*0.2)
    print("=================================================================")