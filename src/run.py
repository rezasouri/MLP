import matplotlib.pyplot as plt
from src.data import X_train, X_test, X_valid, y_test, y_train, y_valid

fig, ax = plt.subplots(10, 10,figsize=(10,10))
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(X_train[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()