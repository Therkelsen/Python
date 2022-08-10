
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

if __name__ == '__main__':
    df = pd.read_csv('auto.csv')
    print('data read from auto.csv')
    print(df.to_string())

    df = df[df['horsepower'] != '?']
    print('cars with a ? in horsepower removed')
    print(df[df['horsepower'] == '?'])

    df['horsepower'] = df['horsepower'].astype(int)
    print('datatype of horsepower column changed to int32')
    print(df.dtypes)

    print('first 5 rows', df.head())

    fig, axis = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    axis[0, 0].scatter(x=df['cylinders'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#ff5555'])
    axis[0, 0].set_xlabel('cylinders')
    axis[0, 0].set_ylabel('mpg')

    axis[0, 1].scatter(x=df['displacement'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#55ff55'])
    axis[0, 1].set_xlabel('displacement')

    axis[0, 2].scatter(x=df['horsepower'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#5555ff'])
    axis[0, 2].set_xlabel('horsepower')

    axis[1, 0].scatter(x=df['weight'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#ffff55'])
    axis[1, 0].set_xlabel('weight')
    axis[1, 0].set_ylabel('mpg')

    axis[1, 1].scatter(x=df['acceleration'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#ff55ff'])
    axis[1, 1].set_xlabel('acceleration')

    axis[1, 2].scatter(x=df['year'], y=df['mpg'], s=10, marker='o', alpha=0.30, c=['#5ffff5'])
    axis[1, 2].set_xlabel('year')

    plt.suptitle('mpg relationships')
    plt.show()

    x = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']].astype(np.float32)
    y = df['mpg'].astype(np.float32)
    plt.show()
    x['year'] = 1/x['year']
    x['acceleration'] = 1/x['acceleration']

    X = x
    Y = y
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('\nR^2: ', model.rsquared)

    # data transformation on x
    X = np.log(x)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('log(X) R^2: ', model.rsquared)

    X = np.sqrt(x)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('sqrt(X) R^2: ', model.rsquared)

    X = 1/x
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('1/X R^2: ', model.rsquared)

    X = np.float_power(x, 2)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('X^2 R^2: ', model.rsquared)

    X = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print('normalized X R^2: ', model.rsquared)

    # data transformation on y
    X = sm.add_constant(X)
    Y = np.log(y)
    model = sm.OLS(Y, X).fit()
    print('\nlog(Y) R^2: ', model.rsquared)

    X = sm.add_constant(X)
    Y = np.sqrt(y)
    model = sm.OLS(Y, X).fit()
    print('sqrt(Y) R^2: ', model.rsquared)

    X = sm.add_constant(X)
    Y = 1 / y
    model = sm.OLS(Y, X).fit()
    print('1/Y R^2: ', model.rsquared)

    X = sm.add_constant(X)
    Y = np.float_power(y, 2)
    model = sm.OLS(Y, X).fit()
    print('Y^2 R^2: ', model.rsquared)

    X = sm.add_constant(X)
    Y = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
    model = sm.OLS(Y, X).fit()
    print('normalized Y R^2: ', model.rsquared)

    # data transformation on x and y
    X = np.log(x)
    X = sm.add_constant(X)
    Y = np.log(y)
    model = sm.OLS(Y, X).fit()
    print('\nlog(x) and log(Y) R^2: ', model.rsquared)

    X = np.sqrt(x)
    X = sm.add_constant(X)
    Y = np.sqrt(y)
    model = sm.OLS(Y, X).fit()
    print('sqrt(X) and sqrt(Y) R^2: ', model.rsquared)

    X = 1 / x
    X = sm.add_constant(X)
    Y = 1 / y
    model = sm.OLS(Y, X).fit()
    print('1/X and 1/Y R^2: ', model.rsquared)

    X = np.float_power(x, 2)
    X = sm.add_constant(X)
    Y = np.float_power(y, 2)
    model = sm.OLS(Y, X).fit()
    print('Y^2 R^2: ', model.rsquared)

    X = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    X = sm.add_constant(X)
    Y = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
    model = sm.OLS(Y, X).fit()
    print('normalized X and Y R^2:', model.rsquared)

    X = np.log(x)
    X = sm.add_constant(X)
    Y = np.log(y)
    model = sm.OLS(Y, X).fit()

    print_model = model.summary()
    print(print_model)
    print(model.params)

    fig1 = sm.graphics.plot_partregress_grid(model)
    fig1.tight_layout(pad=1.0)
    fig1.show()