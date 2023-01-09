import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import argparse
from os import listdir

np.random.seed(42)


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--train', default="house3_5devices_train.csv", required=False, help='Data file')
    parser.add_argument('--test', default="test_folder", required=False, help='Sep. plots?')
    parser.add_argument('--output', default="out", required=False, help='Sep. plots?')

    args = parser.parse_args()
    return args.train, args.test, args.output


data_file, test_dir, out_file = ParseArguments()
lst_l2 = []
lst_l5 = []
lst_l4 = []
lst_rr = []
lst_micro = []
with open(data_file, 'r') as f:
    for line in f:
        line = line.strip('\n')
        time, l2, l5, l4, rr, micro = line.split(',')
        lst_l2.append(l2)
        lst_l5.append(l5)
        lst_l4.append(l4)
        lst_rr.append(rr)
        lst_micro.append(micro)

lst_l2 = [int(lst_l2[i]) for i in range(1, len(lst_l2))]
lst_l5 = [int(lst_l5[i]) for i in range(1, len(lst_l2))]
lst_l4 = [int(lst_l4[i]) for i in range(1, len(lst_l2))]
lst_rr = [int(lst_rr[i]) for i in range(1, len(lst_l2))]
lst_micro = [int(lst_micro[i]) for i in range(1, len(lst_l2))]

mod_l2 = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=100)
mod_l5 = hmm.GaussianHMM(n_components=8, covariance_type="full", n_iter=100)
mod_l4 = hmm.GaussianHMM(n_components=8, covariance_type="full", n_iter=100)
mod_rr = hmm.GaussianHMM(n_components=8, covariance_type="full", n_iter=100)
mod_mic = hmm.GaussianHMM(n_components=7, covariance_type="full", n_iter=100)

mod_l2.fit(np.array(lst_l2).reshape(-1, 1))
mod_l5.fit(np.array(lst_l5).reshape(-1, 1))
mod_l4.fit(np.array(lst_l4).reshape(-1, 1))
mod_rr.fit(np.array(lst_rr).reshape(-1, 1))
mod_mic.fit(np.array(lst_micro).reshape(-1, 1))

scores = []
for i in range(2, 19): #szukamy najlepszych wartości hiperparametru n_components
    mod = hmm.GaussianHMM(n_components=i, covariance_type="full", n_iter=100)
    mod.fit(np.array(lst_rr).reshape(-1, 1))
    scores.append(mod.score(np.array(lst_rr).reshape(-1, 1)))

print([round(i, 3) for i in scores])

#[-60152.676, -54789.361, 35426.73, 37117.881, 42926.498, 38633.943, 38882.541] l2 = 6
#[-59818.827, 42799.673, 49598.96, 50763.907, 55278.97, 56150.454, 57121.603] l5 = 8
#[-68129.204, 21357.169, 26273.679, 30462.289, 33099.27, 34433.535, 36107.371] l4 = 8
#[-38975.491, -27066.248, -26114.633, -23904.608, -23814.112, -23811.998, -22445.653] rr = 8
#[-12937.089, -11131.626, 62293.842, 62307.331, 75962.51, 75970.983, 67809.541] micro = 7

