import numpy as np

class Perceptron:

    def __init__(self, a = 0.1, n_iters = 100):
        self.a = a
        self.n_iters = n_iters

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape

        self.w = np.random.randn(n_features)
        self.b = np.random.rand()

        for i in range(self.n_iters):
            for idx, x in enumerate(X_train):
                pred = self.step_function(np.dot(x, self.w) + self.b)

                self.w += self.a*(y_train[idx] - pred) * x
                self.b += self.a*(y_train[idx] - pred)

        return self.w, self.b

    def predict(self, X_test):
        return self.step_function(np.dot(X_test, self.w) + self.b)

    def step_function(self, y):
        return 1 if y >= 0 else 0

np.random.seed(123)

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = [0, 1, 1, 1]

model = Perceptron()
model.fit(X_train, y_train)
print(model.predict([0, 0]))

