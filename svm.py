import numpy as np


class Svm (object):
    """" Svm classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        deviation = 0.01
        self.W = deviation * np.random.randn(inputDim,outputDim)
        # - Generate a random svm weight matrix to compute loss                 #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################

        pass
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        Svm loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the svm loss and store to loss variable.                        #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                   #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################
        numofclasses = self.W.shape[1]        #get the number of classes
        numoftrain = x.shape[0]               #get the number of training samples
        loss = 0.0
        for i in range(numoftrain):
            # weight matrix W dot products every sample
            scores = x[i].dot(self.W)
            # get the score of the correct class
            correct_class_score = scores[y[i]]
            for j in range(numofclasses):
                if j == y[i]:
                    continue
                # svm loss function
                margin = scores[j] - correct_class_score + 1
                if margin > 0:
                    loss += margin
                    # update the gradients
                    dW[:, j] += x[i]
                    dW[:, y[i]] -= x[i]
        loss /= numoftrain
        dW /= numoftrain
        # add L2 regularization to the loss function
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += 2 * reg * self.W

        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Svm classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            # randomly pick up batchSize samples
            randomIndex = np.random.choice(len(x), batchSize, replace=True)
            x_batch = x[randomIndex]
            y_batch = y[randomIndex]
            # calculate the loss and gradient
            loss,dW = self.calLoss(x_batch,y_batch,reg)
            # update the weight matrix
            self.W = self.W - lr * dW
            lossHistory.append(loss)

            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        result = x.dot(self.W)
        # return the indices of the maximum in each row
        yPred = np.argmax(result, axis = 1)

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        yPred = self.predict(x)
        acc = np.mean(y == yPred)

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



