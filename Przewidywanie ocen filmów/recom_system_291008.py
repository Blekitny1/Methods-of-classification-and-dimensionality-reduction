from sklearn.decomposition import NMF, TruncatedSVD
import argparse
import numpy as np

# Standardowe użycie argparse, umożliwiające łatwe wywoływanie programu z różnymi argumentami z linii poleceń.


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='train.csv', required=True)
    parser.add_argument('--test_file', default='test.csv', required=True)
    parser.add_argument('--alg',  default="NMF", required=True)
    parser.add_argument('--result',  default="result.txt", required=True)
    args = parser.parse_args()
    return args.train_file, args.test_file, args.alg, args.result


def recom_system(train_file='train.csv', test_file='test.csv', alg='NMF', result_file='res.txt'):
    with open(train_file) as f1:
        train_data = []
        test_data = []

        #Musze zebrać gdzieś wszstkie pojawiające się numery użytkowników i filmów, żeby póżniej dla każdego
        #z nich był(a) osobny(a) wiersz lub kolumna macierzy, wygodnie to zrobić w zbiorze, bo w ten sposób
        #na pewno nie pominę żadnego numeru filmu czy użytkownika i nie muszę się zastanawiać jak poradzić
        #sobie z powtarzającymi się numerami. Te zbiory póżniej posortuję i ponumeruję ich elementy słownikiem
        #dzięki czemu bardzo łatwo będzie umieszczać odpowiednie obserwacje w macierzach Z i V.

        user_ids = set()
        movie_ids = set()

        #Odczytuję dane z plików treningowych i testowych, opuszczając pierwszą linjkę
        next(f1)
        for line in f1:
            line = line.strip('\n')
            userId, movieId, rating, timestamp = line.split(',')
            user_ids.add(float(userId))
            movie_ids.add(float(movieId))

            lst = [float(userId), float(movieId), float(rating)]
            train_data.append(lst)

        with open(test_file) as f2:
            next(f2)
            for line in f2:
                line = line.strip('\n')
                userId, movieId, rating, timestamp = line.split(',')
                user_ids.add(float(userId))
                movie_ids.add(float(movieId))

                lst = [float(userId), float(movieId), float(rating)]
                test_data.append(lst)

            user_ids = list(user_ids)
            movie_ids = list(movie_ids)
            user_ids.sort()
            movie_ids.sort()
            n = len(user_ids)
            d = len(movie_ids)
            user_dict = {user_ids[i]: i for i in range(n)}
            movie_dict = {movie_ids[i]: i for i in range(d)}

            Z = np.zeros((n, d)) #Na razie dajmy tam zera
            V = np.zeros((n, d))

            #Słowniki są tak skonstruowane, że do uzupełnienia macierzy Z i V danymi wystarczy pętla po elementach
            #list z danymi.

            for e in train_data:
                row = user_dict[e[0]]
                col = movie_dict[e[1]]
                Z[row, col] = e[2]

            for e in test_data:
                row = user_dict[e[0]]
                col = movie_dict[e[1]]
                V[row, col] = e[2]

        #Funkcja obliczająca RMSE dla dowolnej macierzy wynikowej - porównujemy zawsze z naszymi danymi testowymi,
        #okaże się przydatna przy SVD2.

        def calc_RMSE(A):
            RMSE_squares = 0
            for e in test_data:
                rw = user_dict[e[0]]
                cl = movie_dict[e[1]]
                true_rating = e[2]
                RMSE_squares += (true_rating - A[rw, cl])**2

            return RMSE_squares / len(test_data)

        #Potrzebuję stworzyć obiekt wynikowy Z_prim poza wewnętrznymi instrukcjami warunkowymi

        Z_prim = np.zeros((n, d))

        #Najlepsza wersja NMF - w miejsce zer w macierzy Z wpisujemy średnie oceny filmów dla użytkowników.

        if alg == 'NMF':

            user_avgs = []

            #Obliczamy średnie po wierszach, czyli użytkowników

            for i in range(Z.shape[0]):
                user_nonzero_sum = 0
                user_nonzero_counter = 0
                for j in range(Z.shape[1]):
                    if Z[i, j] > 0:
                        user_nonzero_sum += Z[i, j]
                        user_nonzero_counter += 1

                user_avgs.append(user_nonzero_sum / max(user_nonzero_counter, 1))

            #Uzupełniamy macierz Z w miejscach, gdzie ma zera, tj. nie ma tam danych treningowych.

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    if Z[i, j] == 0:
                        Z[i, j] = user_avgs[i]

            #Wykonujemy NMF

            model = NMF(n_components=22, init='random', random_state=0)
            W = model.fit_transform(Z)
            H = model.components_
            Z_prim = np.dot(W, H)

        #Wykonujemy SVD1

        if alg == 'SVD1':
            svd = TruncatedSVD(n_components=6, random_state=17)
            svd.fit(Z)
            Sigma2 = np.diag(svd.singular_values_)
            VT = svd.components_
            W = svd.transform(Z) / svd.singular_values_
            H = np.dot(Sigma2, VT)
            Z_prim = np.dot(W, H)

        #Ponieważ algorytm SVD2 jest rekurencyjny, potrzebuję drugiego obiektu, w którym będę trzymał poprzednią /
        #następną iterację algorytmu

        if alg == 'SVD2':

            Z_bis = Z
            c = 0

            #Zmienna c jest używana tylko do wejścia do pętli, przez co na początku mogę ustalić Z_bis = Z_prim = Z
            #i póżniej zgodnie z algorytmem modyfikować dwie pierwsze z tych macierzy. Warunek na ograniczenie od dołu
            #różnicy w RMSE między kolejnymi macierzami kończy pracę algorytmu w momencie, w którym następne wyniki
            #różnią się o dostatecznie niewiwle od poprzednich, przez co program działa krócej.

            while abs(calc_RMSE(Z_bis) - calc_RMSE(Z_prim)) > 0.01 or c == 0:

                Z_prim = Z_bis #wynik poprzedniego wykonania staje się argumentem następnego

                for i in range(Z_prim.shape[0]):
                    for j in range(Z_prim.shape[1]):
                        if Z[i, j] > 0:
                            Z_prim[i, j] = Z[i, j] #ustalamy wartości Z_prim tak, żeby były zgodne z danymi treningowymi.

                #Poczym wykonujemy SVD

                svd = TruncatedSVD(n_components=6, random_state=0)
                svd.fit(Z_prim)
                Sigma2 = np.diag(svd.singular_values_)
                VT = svd.components_
                W = svd.transform(Z_prim) / svd.singular_values_
                H = np.dot(Sigma2, VT)
                Z_bis = np.dot(W, H)

                c += 1

        #Na końcu zapisujemy wynik - czyli calc_RMSE(Z_prim) do pliku wynikowego.

        with open(result_file, 'w') as f3:
            f3.write(str(calc_RMSE(Z_prim)))

    return calc_RMSE(Z_prim)


if __name__ == "__main__":

    train_file, test_file, alg, result_file = ParseArguments()
    recom_system(train_file, test_file, alg, result_file)

    #print(recom_system(train_file, test_file, 'NMF', result_file))
    #print(recom_system(train_file, test_file, 'SVD1', result_file))
    #print(recom_system(train_file, test_file, 'SVD2', result_file))