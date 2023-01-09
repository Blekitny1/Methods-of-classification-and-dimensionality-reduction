import json
import numpy as np


# Ponizej prawdziwa wartość macierzy Theta, którą algorytm EM będzie oszacowywał, jest wymiaru 4*w
tmp = np.array([[3/8, 1/8, 2/8, 2/8], [1/10, 2/10, 3/10, 4/10], [1/7, 2/7, 1/7, 3/7], [4/10, 2/10, 3/10, 1/10], [1/9, 2/9, 3/9, 3/9], [3/8, 1/8, 2/8, 2/8], [1/10, 2/10, 3/10, 4/10], [1/7, 2/7, 1/7, 3/7], [4/10, 2/10, 3/10, 1/10], [1/9, 2/9, 3/9, 3/9]])
Theta = tmp.T


# Analogicznie, prawdziwa wartość macierzy Theta_B, którą algorytm EM będzie oszacowywał, wymiaru 4*1
ThetaB = np.array([1/4, 1/4, 1/4, 1/4])

params = {
    "w" : 10,
    "alpha" : 0.8,
    "k" : 1000,
    "Theta" : Theta.tolist(),
    "ThetaB" : ThetaB.tolist()
    }

with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)


#Sprawdzałem czy dobrze działa dla innych parametrów