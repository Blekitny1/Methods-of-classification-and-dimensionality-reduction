from sklearn.model_selection import train_test_split
import numpy as np


def create_train_test(input_file, train_file = 'train.csv', test_file = 'test.csv'):
    with open(input_file) as f:
        data = []
        user_ids = set()
        movie_ids = set()

        #To dlaczego korzystam tutaj ze zbioru jest wyjaśnione w głównym programie recon_system_291008.
        #Deleted_col_names potrzebuję tylko po to, żeby nie zapisywać danych z pierwszej linii pliku

        next(f)
        for line in f:
            line = line.strip('\n')

            userId, movieId, rating, timestamp = line.split(',')
            user_ids.add(int(userId))
            movie_ids.add(int(movieId))

            lst = [int(userId), int(movieId), float(rating), int(timestamp)]
            data.append(lst)

        user_ids = list(user_ids)
        movie_ids = list(movie_ids)
        user_ids.sort()
        movie_ids.sort()

        #Tworzę obiekty typu np.array z jednym wierszem z przypadkowymi liczbami, ponieważ w pętli bedę
        #'doklejać' do nich tablice zawierające około 90% lub 10% ocen filmów każdego użytkownika. Po przebiegu
        #pętli pierwszy wiersz macierzy usuwam, a nastęnie wklejam macierze do plików.

        train_arr = np.array([17, 17, 17, 17])
        test_arr = np.array([17, 17, 17, 17])
        n = len(data)

        for UserId in user_ids:
            user_arr = np.array([data[i] for i in range(n) if data[i][0] == UserId])
            user_train, user_test = train_test_split(user_arr, test_size=0.1)
            train_arr = np.vstack((train_arr, user_train))
            test_arr = np.vstack((test_arr, user_test))

        train_arr = np.delete(train_arr, 0, axis=0)
        test_arr = np.delete(test_arr, 0, axis=0)

        with open(train_file, 'w') as f1:
            f1.write('userId,movieId,rating,timestamp\n')
            for i in range(train_arr.shape[0]):
                lst = [str(train_arr[i, 0]), str(train_arr[i, 1]), str(train_arr[i, 2]), str(train_arr[i, 3])]
                f1.write(','.join(lst) + '\n')

        with open(test_file, 'w') as f2:
            f2.write('userId,movieId,rating,timestamp\n')
            for i in range(test_arr.shape[0]):
                lst = [str(test_arr[i, 0]), str(test_arr[i, 1]), str(test_arr[i, 2]), str(test_arr[i, 3])]
                f2.write(','.join(lst) + '\n')


create_train_test('ratings_0.csv', 'train_0.csv', 'test_0.csv')