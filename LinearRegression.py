import matplotlib as plt
import numpy as np

class LinearRegression:
    def __init__(self,X,y):
        self.sampleCount=len(y)
        self.featureCount=len(X[0])+1
        self.X = np.ones((self.sampleCount,self.featureCount))
        self.X[:,:-1]=X
        self.params=np.zeros((3,1))
        self.y=y
    def gradientDescent(self,iterations=int(1e4),learningRate=1e-4):
        def MSE_partialDerivatives(X,y,weights):
            scaleFactor = -2/len(y)
            firstTerm = X.T
            secondTerm=y-X@weights
            return scaleFactor*firstTerm@secondTerm
        def MSE(X,y,weights):
            """ Compute mean square error for linear function

            weights
            ------
            X : matrix (sample size,number of features)
                Feature matrix
            y : column vector (sample size, 1)
                observed values
            weights : column vector (number of features,1)
                parameters of linear function
            """
            sampleSize=len(y)
            yPredicted = X @ weights #Use inproduct to calculate predicted values
            MSE = np.sum((y-yPredicted)**2)/sampleSize
            return MSE
        self.costEvolution=np.zeros(iterations)
        self.paramsEvoltion=np.zeros((iterations,len(self.params)))
        for i in range(iterations):
            self.params = self.params - learningRate*MSE_partialDerivatives(self.X,self.y,self.params)
            self.paramsEvoltion[i,:]=self.params[:,0]
            self.costEvolution[i]=MSE(self.X,self.y,self.params)
        return self.params, self.paramsEvoltion,self.costEvolution
    def plotGradientDescent(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.costEvolution)
        plt.title("Evolution of cost function during Gradient descent")
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.show()
        
        plt.figure(figsize=(14,7))
        for i in range(len(self.paramsEvoltion[0])):
            plt.plot(self.paramsEvoltion[:,i],label="parameter {}".format(i))
        plt.title("Evolution of cost function during Gradient descent")
        plt.xlabel("iteration")
        plt.ylabel("parameter value")
        plt.legend()
        plt.show()