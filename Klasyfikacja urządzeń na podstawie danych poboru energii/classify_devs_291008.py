import numpy as np
from hmmlearn import hmm
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
mod_rr = hmm.GaussianHMM(n_components=18, covariance_type="full", n_iter=100)
mod_mic = hmm.GaussianHMM(n_components=7, covariance_type="full", n_iter=100)

mod_l2.fit(np.array(lst_l2).reshape(-1, 1))
mod_l5.fit(np.array(lst_l5).reshape(-1, 1))
mod_l4.fit(np.array(lst_l4).reshape(-1, 1))
mod_rr.fit(np.array(lst_rr).reshape(-1, 1))
mod_mic.fit(np.array(lst_micro).reshape(-1, 1))

#scores = []
#for i in range(2, 9):
#    mod = hmm.GaussianHMM(n_components=i, covariance_type="full", n_iter=100)
#    mod.fit(np.array(lst_micro).reshape(-1, 1))
#    scores.append(mod.score(np.array(lst_micro).reshape(-1, 1)))
#
#print([round(i, 3) for i in scores])

with open(out_file, 'w') as of:
    of.write('file, dev. classified\n')
    for filename in listdir(test_dir):
        with open(test_dir+'/'+filename, 'r') as f:
            ls = []
            for line in f:
                line = line.strip('\n')
                time, val = line.split(',')
                ls.append(val)

            ls = [int(ls[i]) for i in range(1, len(ls))]
            ls = np.array(ls).reshape(-1, 1)
            scores = [mod_l2.score(ls), mod_l5.score(ls), mod_l4.score(ls), mod_rr.score(ls), mod_mic.score(ls)]
            print([round(i, 3) for i in scores])
            winner = scores.index(max(scores)) #wybieramy model, z najwieksza wierygodnością
            if winner == 0:
                of.write(str(filename) + ', lighting2\n')
            elif winner == 1:
                of.write(str(filename) + ', lighting5\n')
            elif winner == 2:
                of.write(str(filename) + ', lighting4\n')
            elif winner == 3:
                of.write(str(filename) + ', refrigerator\n')
            else:
                of.write(str(filename) + ', microwave\n')


