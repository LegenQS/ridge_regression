import csv
import os
import numpy as np
import matplotlib.pyplot as plt

class Ridge:
    def __init__(self, file_name, lamb, w_label=None, w_RR=None, X_test=None, y_test=None):
        # initialize all the paramters needed within class Ridge
        # file_name: path of file to read
        # lamb: user defined value of λ
        # w_label: label of dimensions corresponding to the weight of ridge regression
        # w_RR: the values of weight by ridge regression
        # X_test: only used in question three to test different orders
        # y_test: only used in question three to test different orders

        self.X = self.read_file(file_name[0])
        self.y = self.read_file(file_name[1])
        self.X_test = X_test
        self.y_test = y_test
        self.lamb = lamb
        self.length = self.X.shape[1]
        self.w_RR = np.empty((self.lamb, self.length))
        if w_RR is not None:
            self.w_RR = w_RR
        self.w_label = w_label

    def read_file(self, filename):
        # read file with fiven filename and return nparray format
        # parameter lists:
        # ------------------------------------------------------
        # filename: path of the file to read

        return np.loadtxt(open(filename, "r"), delimiter=",", skiprows=0, dtype = float)

    def wRR(self, length=0, X=None):
        # calculate the value of weights by ridge regression inner class or with given parameter
        # parameter lists:
        # ------------------------------------------------------
        # length: the total number of columns with given X or within inner class
        # X: given train set sample

        if X is None:
            for i in range(self.lamb):
                self.w_RR[i,:] = np.dot(np.dot(np.linalg.inv(np.diag([i] * self.length) +
                                        np.dot(self.X.T, self.X)), self.X.T), self.y).reshape(self.length, )
        else:
            w_RR = np.empty((self.lamb, length))
            for i in range(self.lamb):
                w_RR[i,:] = np.dot(np.dot(np.linalg.inv(np.diag([i] * length) +
                                        np.dot(X.T, X)), X.T), self.y).reshape(length, )
            return w_RR

    def SVD(self, lamb):
        # use singular value decomposition to calculate df(λ)
        # parameter lists:
        # ------------------------------------------------------
        # lamb: the value of λ

        U, S, VT = np.linalg.svd(self.X)
        df = 0
        for j in S:
            df += pow(j,2) / (lamb + pow(j, 2))
        return df

    def Miu_V(self, order, sign=0):
        # construct new sample set with given order, normalize each added column by standardization with miu and sigma
        # parameter lists:
        # ------------------------------------------------------
        # order: the exponential parameter we want to expand
        # sign: use to generate train_set with 0 or test_set with 1

        corr = self.X[:,0:6]
        if sign:
            add = self.X_test[:, 0:6]
            result = self.X_test
        else:
            add = self.X[:, 0:6]
            result = self.X

        i = 1
        while(i - order):
            miu = np.mean(pow(corr, i + 1), axis=0)
            sigma = np.std(pow(corr, i + 1), axis=0)
            result = np.append(result, (pow(add, i + 1) - miu) / sigma, axis=1)
            i += 1

        return result

    def RMSE(self, X=None, w_RR=None):
        # calculate RMSE, if not given parameters, calculate with inner class parameters, else using given parameters
        # parameter lists:
        # ------------------------------------------------------
        # X: test_set samples
        # w_RR: values of weights using ridge regression
        if w_RR is None:
            return np.sqrt(np.mean(pow((np.dot(self.X, self.w_RR.T) - self.y.reshape(-1, 1)), 2), axis=0))
        else:
            return np.sqrt(np.mean(pow((np.dot(X, w_RR.T) - self.y_test.reshape(-1, 1)), 2), axis=0))

    def myplot(self, x, y, fig_name=None, plot_label=None, x_label='x', y_label='y', line_label=[""], title=None, seq=0):
        # user defined plot
        # parameter lists:
        # ------------------------------------------------------
        # x: first parameter of plt.plot on x-axis, assuming only one dimension by default
        # y: second parameter of plt.plot on y-axis, with any user defined dimension as long as suitable for x
        # fig_name: the name of plot figure when save figure to local
        # x_label: the title of x-axis label, 'x' by default
        # y_label: the title of y-axis label, 'y' by default
        # line_label: when multiply y input, define each line with its own label (list format required)
        # title: the title of the figure, None by default
        # seq: sequence number to distinguish plot as a whole or plot column or row of y each time with 0 and not by 1

        plt.figure(figsize=(50, 30))
        for plot_y in range(len(y)):
            if not seq:
                for i in range(self.length):
                    plt.plot(x, y[plot_y][:, i], label=plot_label % i, linewidth=3)
            else:
                plt.plot(x, y[plot_y], label=line_label[plot_y], linewidth=3)

        plt.tick_params(labelsize=55)
        plt.xlabel(x_label, fontsize=55)
        plt.ylabel(y_label, fontsize=55)
        if title:
            plt.title(title, fontsize=55)
        if plot_label or len(y) > 1:
            plt.legend(fontsize=55)
        plt.savefig(fig_name)
        plt.show()
        plt.close()

    def question_one(self):
        df_lambda = []
        self.wRR()
        for i in range(self.lamb):
            df_lambda.append(self.SVD(i))

        self.myplot(df_lambda, [self.w_RR], fig_name='1-1.png', plot_label='$\omega_{RR% d}$', title='Relationship' + \
        'between $df(\lambda)$ and attributes ', x_label='value of $df(\lambda)$', y_label='value of $\omega$', seq=0)

    def question_two(self):
        test_lambda = np.linspace(0, self.lamb-1, self.lamb)
        RMSE = self.RMSE()

        self.myplot(test_lambda, [RMSE], fig_name='1-2.png', x_label='value of $\lambda$',
                  y_label='value of RMSE', title='relationship between the value of $\lambda$ and RMSE', seq=1)

    def question_three(self, order=1):
        test_lambda = np.linspace(0, self.lamb-1, self.lamb)
        if order > 1:
            X_train = [None] * order
            X_test = [None] * order
            wRR = [None] * order
            RMSE = [None] * order
        for i in range(order):
            if not i:
                X_train[i] = self.X
                X_test[i] = self.X_test
                wRR[i] = self.w_RR
                RMSE[i] = self.RMSE(X=self.X_test, w_RR=wRR[i])
            else:
                X_train[i] = self.Miu_V(i+1)
                X_test[i] = self.Miu_V(i+1, sign=1)
                wRR[i] = self.wRR(X_train[i].shape[1], X_train[i])
                RMSE[i] = self.RMSE(X=X_test[i], w_RR=wRR[i])

        self.myplot(test_lambda, RMSE, fig_name='1-3.png', x_label='value of $\lambda$',
                    y_label='value of RMSE', line_label=['1-th order', '2-th order', '3-th order'], seq=1)

def main():
    # user defined function to achieve different functions
    # train_file, test_file: local path of train set files and test set files
    # lamb_one, lamb_two, lamb_three: user defined lambda on different functions
    # w_label: given dimension labels of weight of ridge regression
    # order: user defined polynomial regression order

    train_file = ["./X_train.csv", "./y_train.csv"]
    lamb_one = 5001
    w_label = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year made", "intercept"]
    test_file = ["./X_test.csv", "./y_test.csv"]
    lamb_two = 51
    lamb_three = 101

    # function one
    ridge_train = Ridge(train_file, w_label=w_label, lamb=lamb_one)
    ridge_train.question_one()

    # function two
    ridge_test = Ridge(test_file, lamb=lamb_two, w_RR=ridge_train.w_RR[:lamb_two])
    ridge_test.question_two()

    # function three
    ridge_three = Ridge(train_file, lamb=lamb_three, w_RR=ridge_train.w_RR[:lamb_three],
                        X_test=ridge_test.X, y_test=ridge_test.y)
    ridge_three.question_three(order=3)

if __name__ == '__main__':
    main()
