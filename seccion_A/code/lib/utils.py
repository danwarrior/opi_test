from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')


def display_grid(mode, trends, sorted_array, delitos_date):
    assert mode in ['up', 'down'], 'Mode must be \'up\' or \'down\''

    if mode == 'up':
        delitos = sorted_array[:5]
    elif mode == 'down':
        delitos = np.flip(sorted_array[-5:])

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    t = 0
    for i in range(2):
        for j in range(2):
            # print(t)
            t += 1

            delito = trends[delitos[t]]
            data = delitos_date[delito]

            y = data.values
            x = np.arange(0, len(data.index)).reshape(-1, 1)
            reg = LinearRegression().fit(x, y)

            ax[i, j].scatter(x, y)
            ax[i, j].plot(x, reg.coef_ * x + reg.intercept_, c='r', lw=5)
            ax[i, j].set_title(f'{delito[:20]}, trend: {reg.coef_[0]:.4f}')
            ax[i, j].set_xticklabels(list(map(lambda x: str(x)[:10], data.index)), rotation=45)
    fig.tight_layout()
    plt.show()
