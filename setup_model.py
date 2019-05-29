import numpy as np
import cv2
import matplotlib.pyplot as plt 


class LogisticRegression:
    """
        set up from scatch Logistic Regression model
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def predict(self, x):
        """
            predict results
        """
        z = np.sum(x * self.W, axis=1)
        return 1 / (1 + np.exp(-z))

    def learn(self, x, y, learning_rate):
        y_hat = self.predict(x)
        new_W = np.matmul(x.T, y - y_hat)
        self.W = self.W + learning_rate * new_W
        
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        return {
            "loss": -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)),
            "accuracy": np.sum((y_pred > self.threshold).astype(int) == y) / y.shape[0]
        }
    
    def fit(
        self, x, y, x_valid = None, y_valid = None,
        learning_rate = 0.001,
        learning_rate_decay = 1,
        batch_size = 32,
        epoch = 1,
        verbose = False
    ):
        self.W = np.random.rand(x.shape[1])
        if x_valid is None:
            x_valid = x
        if y_valid is None:
            y_valid = y
        step = x.shape[0] // batch_size + (x.shape[0] % 2 == 0)
        metric_graph = {
            "loss": [],
            "accuracy": []
        }
        for e in range(epoch):
            for i in range(step):
                self.learn(
                    x[batch_size * i : batch_size * (i + 1),],
                    y[batch_size * i : batch_size * (i + 1),],
                    learning_rate
                )
                metrics = self.evaluate(x_valid, y_valid)
                if (e <= 5 or (i + 1) == step) and verbose:
                    metrics = self.evaluate(x_valid, y_valid)
                    print("Epoch %d Step %d: Loss %f, Acc %f" % (e + 1, i + 1, metrics["loss"], metrics["accuracy"]))
            
            metrics = self.evaluate(x_valid, y_valid)
            metric_graph["loss"].append(metrics["loss"])
            metric_graph["accuracy"].append(metrics["accuracy"])
            learning_rate *= learning_rate_decay
        
        plt.plot(metric_graph["loss"])
        plt.title("Loss")
        plt.show()
        plt.title("Accuracy")
        plt.plot(metric_graph["accuracy"])
        plt.show()
