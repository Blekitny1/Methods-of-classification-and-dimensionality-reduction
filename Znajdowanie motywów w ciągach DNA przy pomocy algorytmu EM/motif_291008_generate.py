import json
import argparse
from random import choices


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as input_file:
    params = json.load(input_file)
 
 
w = params['w']
k = params['k']
alpha = params['alpha']
Theta = params['Theta']
distributions = [[Theta[i][j] for i in range(4)] for j in range(w)]
ThetaB = params['ThetaB']

#Do uzyskania rozkładów, czyli wierszy macierzy Theta wygodnie użyć list składanych

#wybieram w ktorych wierszach uzywam thety a w których thety_B zgodnie z wartością parametru alfa, to samo oznaczenie co w opisie projektu
z = choices([0, 1], [1 - alpha, alpha], k=k)

Xs = []
for i in range(k):
    if z[i] == 0:
        Xs.append(choices([1, 2, 3, 4], ThetaB, k=w))
    if z[i] == 1:
        Xs.append([choices([1, 2, 3, 4], distributions[j], k=1)[0] for j in range(w)])

gen_data = {    
    "alpha": alpha,
    "X": Xs
    }

with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)

