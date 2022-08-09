import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # exercise 1
    a = np.full((2, 3), 4)
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = np.eye(2, 3)
    d = a + b + c

    print('\na: \n', a)
    print('\nb: \n', b)
    print('\nc: \n', c)
    print('\nd: \n', d)

    a = np.array([[1, 2, 3, 4, 5],
                  [5, 4, 3, 2, 1],
                  [6, 7, 8, 9, 0],
                  [0, 9, 8, 7, 6]])

    print('\nsum of rows of a:\n', a.sum(axis=1))
    print('\ntranspose of a:\n', a.transpose())

    # exercise 2
    print('data read from auto.csv')
    df = pd.read_csv('auto.csv')
    print(df.to_string())

    df = df[df['mpg'] >= 16]
    print('cars with mpg < 16 removed')
    print(df[df['mpg'] < 16])

    print(df[0:7][['weight', 'acceleration']])

    df = df[df['horsepower'] != '?']
    print('cars with a ? in horsepower removed')
    print(df[df['horsepower'] == '?'])

    print('datatype of horsepower column changed to int32')
    df['horsepower'] = df['horsepower'].astype(int)
    print(df.dtypes)

    print(df.select_dtypes(include='number').mean())
    # exercise 3
    a = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34])
    b = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1])

    data = [a,
            b]

    plt.plot(data[0], label='training accuracy')
    plt.plot(data[1], label='validation accuracy')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.show(block=False)
    plt.close()
