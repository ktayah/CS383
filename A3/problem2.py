import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

EPSILON = 2 ** -23

def main():
    epochs = 0
    J = 0
    l = 0.01
    w = np.zeros(2)
    X = np.ones(2)

    j_values = [J]
    w_values = [w]

    while True:
        D_w0 = (2 * X[0]) * (X[0]*w[0] - 5*X[1]*w[1] - 2)
        D_w1 = (-10 * X[1]) * (X[0]*w[0] - 5*X[1]*w[1] - 2)

        w0 = w[0] - l * D_w0
        w1 = w[1] - l * D_w1
        w = np.array([w0, w1])
        new_J = (X[0] * w[0] - 5 * X[1] * w[1] - 2)**2

        j_values.append(new_J)
        w_values.append(w)

        epochs += 1
        if abs(new_J - J) < EPSILON:
            J = new_J
            break
        else:
            J = new_J

    # J vs epoch
    plt.plot(list(range(0, epochs + 1)), j_values)
    plt.title('Epoch vs J')
    plt.xlabel('epochs')
    plt.ylabel('J')
    plt.show()

    # w1 vs w2 vs J
    fig = plt.figure()
    w1 = np.array(w_values)[:,0]
    w2 = np.array(w_values)[:,1]
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('J')
    ax.plot(w1, w2, zs=j_values)
    plt.show()

    print('w:', w)
    print('J', J)
    print('epochs', epochs)

if __name__ == '__main__':
    main()