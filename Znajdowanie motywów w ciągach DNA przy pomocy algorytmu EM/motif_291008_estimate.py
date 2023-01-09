import json
import numpy as np
import argparse 


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='Plik z danymi')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Tutaj zapiszemy wyestymowane parametry')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha
    
    
input_file, output_file, estimate_alpha = ParseArguments()
 
with open(input_file, 'r') as file:
    data = json.load(file)

alpha = data['alpha']
X = np.asarray(data['X'])
k, w = X.shape

# Theta0 = wektor rozmiaru w
# Theta = macierz rozmiaru d = 4 na w
# przyklad losowy - niech to będą początkowe thety
ThetaB = np.array([1/4, 1/4, 1/4, 1/4])
Theta = np.ones((4, w)) * 1/4 #ok

def pstwo_xi_od_thety(xi, theta):
    r = 1
    for l in range(w):
        r *= theta[int(xi[l]) - 1, l] #jedyna poprawka jest tutaj
    return r


def pstwo_xi_od_thetyb(xi, thetab):
    r = 1
    for l in range(w):
        r *= thetab[int(xi[l]) - 1 #tu analogiczna poprawka
    return r

nowa_theta = 1
nowa_theta_b = [1, 2, 3, 4]
nr_iter = 0
while max([abs(nowa_theta_b[i] - ThetaB[i]) for i in range(4)]) > 0.00001: #przerwij, gdy algorytm zbiegł

    if nr_iter:
        Theta = nowa_theta
        ThetaB = nowa_theta_b

    ile_jedynek_na_miejscu = [0 for _ in range(w)]#dla Z=1 tak jak w przykładzie z monetami zliczam estymowane
    ile_dwojek_na_miejscu = [0 for _ in range(w)]#prawdopodobienstwa każdej z cyferek, na końcu wystarczy będzie
    ile_trojek_na_miejscu = [0 for _ in range(w)]#przeskalować listy, aby suma każdej kolumny macierzy wynosiła 1
    ile_czworek_na_miejscu = [0 for _ in range(w)]#aby otrzymać następną macierz \Theta
    ile_jedynek_b = 0#dla Z=0 nie musimy zliczać osobno dla każdego miejsca, ponieważ w tym przypadku rozkład
    ile_dwojek_b = 0#na każdym miejscu jest ten sam, zatem zliczamy tylko prawdopodobieństwa każdej z cyferek,
    ile_trojek_b = 0#pod koniec skalujemy by otrzymać rozkład \Theta^b
    ile_czworek_b = 0

    for i in range(k):
        wiersz = X[i]
        p_theta = alpha * pstwo_xi_od_thety(wiersz, Theta)
        p_thetab = (1 - alpha) * pstwo_xi_od_thetyb(wiersz, ThetaB)
        Qi_0 = p_thetab / (p_theta + p_thetab) #wiem już, którą "monetą" rzucam z jakim prawdopodobieństwem
        Qi_1 = p_theta / (p_theta + p_thetab)

        for j in range(w):
            if wiersz[j] == 1:
                ile_jedynek_na_miejscu[j] += Qi_1
                ile_jedynek_b += Qi_0
            if wiersz[j] == 2:
                ile_dwojek_na_miejscu[j] += Qi_1
                ile_dwojek_b += Qi_0
            if wiersz[j] == 3:
                ile_trojek_na_miejscu[j] += Qi_1
                ile_trojek_b += Qi_0
            if wiersz[j] == 4:
                ile_czworek_na_miejscu[j] += Qi_1
                ile_czworek_b += Qi_0

    suma_b = ile_jedynek_b + ile_dwojek_b + ile_trojek_b + ile_czworek_b
    nowa_theta_b = [ile_jedynek_b/suma_b, ile_dwojek_b/suma_b, ile_trojek_b/suma_b, ile_czworek_b/suma_b]
    nowa_theta_b = np.array(nowa_theta_b)

    czynnik_skalujacy = ile_jedynek_na_miejscu[0] + ile_dwojek_na_miejscu[0] + ile_trojek_na_miejscu[0] + ile_czworek_na_miejscu[0]
    #to jest rowne tyle samo dla kazdego indeksu tych list i wynosi suma Qi_1, zatem jak podzielimy wszystkie listy
    #przez te liczbe i złożymy w tablicę, to dostaniemy kolejną estymację Thety

    przeskalowana1 = [i/czynnik_skalujacy for i in ile_jedynek_na_miejscu]
    przeskalowana2 = [i/czynnik_skalujacy for i in ile_dwojek_na_miejscu]
    przeskalowana3 = [i/czynnik_skalujacy for i in ile_trojek_na_miejscu]
    przeskalowana4 = [i/czynnik_skalujacy for i in ile_czworek_na_miejscu]

    nowa_theta = np.asarray([przeskalowana1, przeskalowana2, przeskalowana3, przeskalowana4])
    nr_iter += 1

#prawdziwe wartości parametrów Theta i ThetaB, potrzebne do sprawdzenia dokładności oszacowania
tmp = np.array([[3/8, 1/8, 2/8, 2/8], [1/10, 2/10, 3/10, 4/10], [1/7, 2/7, 1/7, 3/7], [4/10, 2/10, 3/10, 1/10], [1/9, 2/9, 3/9, 3/9], [3/8, 1/8, 2/8, 2/8], [1/10, 2/10, 3/10, 4/10], [1/7, 2/7, 1/7, 3/7], [4/10, 2/10, 3/10, 1/10], [1/9, 2/9, 3/9, 3/9]])
Theta_true = tmp.T
ThetaB_true = np.array([1/4, 1/4, 1/4, 1/4])


def goodness(): #funkcja do mierzenia jak mało / dużo algorytm się pomylił
    r = 0
    for i in range(4):
        for j in range(w):
            r += abs(Theta_true[i][j] - nowa_theta[i][j])
        r += abs(ThetaB_true[i] - nowa_theta_b[i])
    return r

#print('gs:' + str(goodness()))

estimated_params = {
    "alpha": alpha,            # "przepisujemy" to alpha, ono nie bylo estymowane
    "Theta": Theta.tolist(),   # wyestymowane
    "ThetaB": ThetaB.tolist()  # wyestymowane
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    
    
    
