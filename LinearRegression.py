import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:
    def __init__(self,X,y):
        self.sampleCount=len(y)
        self.featureCount=len(X[0])+1
        self.X = np.ones((self.sampleCount,self.featureCount))
        self.X[:,:-1]=X
        self.params=np.zeros((self.featureCount,1))
        self.y=y
        self.costEvolution=np.zeros(0)
        self.paramsEvoltion=np.zeros((0,self.featureCount))
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
        start=len(self.costEvolution)
        self.costEvolution=np.concatenate((self.costEvolution,np.zeros(iterations)))
        self.paramsEvoltion=np.concatenate((self.paramsEvoltion, np.zeros((iterations,len(self.params)))))
        for j in range(iterations):
            i =j+start
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
        xlim = (int(len(self.costEvolution)/2),len(self.costEvolution)-1)
        ylim = (self.costEvolution[xlim[1]],self.costEvolution[xlim[0]])
        #plt.ylim(ylim)
        #plt.xlim(xlim)
        plt.show()
        
        plt.figure(figsize=(14,7))
        for i in range(len(self.paramsEvoltion[0])):
            plt.plot(self.paramsEvoltion[:,i],label="parameter {}".format(i))
        plt.title("Evolution of cost function during Gradient descent")
        plt.xlabel("iteration")
        plt.ylabel("parameter value")
        plt.legend()
        plt.show()
    def predictTestSet(self,X_test,y_test):
        X = np.ones((len(y_test),self.featureCount))
        X[:,:-1] = X_test
        y_predicted = X @ self.params
        """ Multiply predicted with known values
        Correctly predicted values are positive
        wrongly predicted are negative
        """
        y_correct_false = y_predicted * y_test
        ratio_correct = len(y_correct_false[y_correct_false>0])/len(y_correct_false)
        return ratio_correct