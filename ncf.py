import gc
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.objectives import mean_squared_logarithmic_error

import grouplen

TRAIN_PATH = "data/grouplens/ua.base"
TEST_PATH = "data/grouplens/ua.test"


def create_model(dim):
    _model = Sequential()
    _model.add(Dense(input_shape=(dim,), output_dim=16))
    _model.add(Dense(16, activation='relu'))
    _model.add(Dense(32, activation='relu'))
    _model.add(Dense(1, activation='relu'))
    return _model


def objective_function(y_true, y_pred):
    return mean_squared_logarithmic_error(y_true, y_pred)


def create_matrix(users, items, rates):
    a = np.zeros(shape=(len(users), len(items)))
    for i in range(0, len(users)):
        a[users[i], items[i]] = rates[i]

    return a


def create_vectors(users, items, rates):
    inputs = []
    labels = []
    co = 0

    num_non_zero = len(rates)
    num_zero = int(num_non_zero * 2)

    co_zero = 0
    mat = create_matrix(users, items, rates)
    for i in range(len(users)):
        for j in range(len(items)):
            co += 1
            rate = mat[i, j]

            u_vec = np.zeros(len(users))
            u_vec[users.index(i)] = 1

            i_vec = np.zeros(len(items))
            i_vec[items.index(j)] = 1

            if rate > 0:
                inputs.append(_get_layer_vector(u_vec, i_vec))
                labels.append(_get_label(rate))
            elif random.randint(0, 1) == 1 and co_zero < num_zero:
                inputs.append(_get_layer_vector(u_vec, i_vec))
                labels.append(_get_label(0))
                co_zero += 1

            if co % 300 == 0:
                yield (np.array(inputs), np.array(labels))
                inputs = []
                labels = []

            if num_zero + num_non_zero - co <= 300:
                return


def create_vectors_test(users, items, rates):
    inputs = []
    labels = []
    co = 0
    for rate in rates:
        co += 1
        user_id = rate[0]
        item_id = rate[1]
        rate = rate[2]

        u_vec = np.zeros(len(users))
        u_vec[users.index(user_id)] = 1

        i_vec = np.zeros(len(items))
        i_vec[items.index(item_id)] = 1

        inputs.append(_get_layer_vector(u_vec, i_vec))
        labels.append(_get_label(rate))

    return np.array(inputs), np.array(labels)


def _get_label(rate):
    if rate > 0:
        return 1
    else:
        return 0


def _get_layer_vector(u_vec, i_vec):
    v = np.append(u_vec, i_vec)
    return v


if __name__ == "__main__":
    _users_train, _items_train, _rates_train = grouplen.read_one(TRAIN_PATH)
    _users_test, _items_test, _rates_test = grouplen.read_one(TEST_PATH)

    _users = [x for x in np.unique(np.append(_users_train, _users_test)).tolist()]
    _items = [x for x in np.unique(np.append(_items_train, _items_test)).tolist()]

    # training and testing
    model = create_model(len(_users) + len(_items))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    test_data = create_vectors_test(_users, _items, _rates_test)
    print(len(test_data[0]))
    gc.collect()
    model.fit_generator(generator=create_vectors(_users, _items, _rates_train),
                        samples_per_epoch=3000, nb_val_samples=300, nb_epoch=20,
                        validation_data=create_vectors(_users, _items, _rates_test))

    score = model.evaluate(test_data[0], test_data[1])
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
