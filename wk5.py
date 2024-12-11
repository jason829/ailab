import numpy as np
from random import random

class Full_NN:
    def __init__(self, X, HL, Y):
        self.X = X
        self.HL = HL
        self.Y = Y

        L = [X] + HL + [Y]  # Network structure

        # Weight initialization using Xavier Initialization
        self.W = [np.random.randn(L[i], L[i + 1]) * np.sqrt(2 / (L[i] + L[i + 1])) for i in range(len(L) - 1)]
        self.B = [np.zeros((1, L[i + 1])) for i in range(len(L) - 1)] 
        self.out = [np.zeros((1, layer)) for layer in L]
        self.Der = [np.zeros_like(w) for w in self.W] 

    def FF(self, x):
        self.out[0] = x.reshape(1, -1)
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            Xnext = np.dot(self.out[i], w) + b
            if i < len(self.W) - 1:
                self.out[i + 1] = self.ReLU(Xnext)  # ReLU for hidden layers
            else:
                self.out[i + 1] = self.sigmoid(Xnext)  # Sigmoid for the output layer
        return self.out[-1]

    def BP(self, Er, lr):
        for i in reversed(range(len(self.Der))):
            out = self.out[i + 1]
            if i == len(self.Der) - 1:
                delta = Er * self.sigmoid_Der(out) # Outer layer
            else:
                delta = Er * self.ReLU_Der(out)  # Hidden layers

            self.Der[i] = np.dot(self.out[i].T, delta)
            self.B[i] = delta.sum(axis=0, keepdims=True)
            
            self.W[i] += self.Der[i] * lr
            self.B[i] += self.B[i] * lr
            
            Er = np.dot(delta, self.W[i].T)

    def train_nn(self, x, target, epochs, lr):
        error_per_epoch = []
        for epoch in range(epochs):
            total_error = 0
            for j, input in enumerate(x):
                t = target[j].reshape(1, -1)
                output = self.FF(input)
                error = t - output
                total_error += self.msqe(t, output)
                self.BP(error,lr)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Error: {total_error / len(x)}")
            
            error_per_epoch.append(total_error/len(x))
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(error_per_epoch) + 1), error_per_epoch, marker='o', label="MAE")
        plt.title("Mean Absolute Error Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.legend()
        plt.savefig("part_d_plot.png")
        plt.show()
        

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_Der(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_Der(self, x):
        return x * (1 - x)

    def msqe(self, t, output):
        return np.mean((t - output) ** 2)

# Training data: Teach multiplication
training_inputs = np.array([[random() for _ in range(2)] for _ in range(1000)])
targets = np.array([[i[0] * i[1]] for i in training_inputs])

nn = Full_NN(2, [10, 10], 1)  # 2 inputs, 2 hidden layers with 10 neurons each, 1 output
nn.train_nn(training_inputs, targets, 1000, 0.005)  # Train for 500 epochs with lr = 0.01

test_input = np.array([0.5, 0.2])
target = np.array([0.5 * 0.2])
output = nn.FF(test_input)

print("=============== Testing the Network Output ===============")
print(f"Test input: {test_input}")
print(f"Target output: {target}")
print(f"Neural Network output: {output}")
print(f"Error: {target - output}")

