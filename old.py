import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.objectives import mean_squared_logarithmic_error

import grouplen

TRAIN_PATH = "data/grouplens/ua.base"
TEST_PATH = "data/grouplens/ua.test"


def create_model(dim):
    _model = Sequential()
    _model.add(Dense(input_shape=(dim,), output_dim=20, activation='relu'))
    _model.add(Dense(20, activation='relu'))
    _model.add(Dense(20, activation="relu"))
    _model.add(Dense(input_dim=20, output_dim=20, activation="relu"))
    _model.add(Dense(input_dim=20, output_dim=1))
    return _model


def objective_function(y_true, y_pred):
    return mean_squared_logarithmic_error(y_true, y_pred)


def _create_vectors(users, items, rates):
    inputs = []
    labels = []
    for rate in rates:
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
    return 0 if rate == 0 else 1


def _get_layer_vector(u_vec, i_vec):
    return np.concatenate([u_vec, i_vec])


if __name__ == "__main__":
    _users_train, _items_train, _rates_train = grouplen.read_one(TRAIN_PATH)
    _users_test, _items_test, _rates_test = grouplen.read_one(TEST_PATH)

    _users = [x for x in np.unique(np.append(_users_train, _users_test)).tolist()]
    _items = [x for x in np.unique(np.append(_items_train, _items_test)).tolist()]

    x_train, y_train = _create_vectors(_users, _items, _rates_train)
    x_test, y_test = _create_vectors(_users, _items, _rates_test)

    print(x_train.shape[1:])

    # training and testing
    model = create_model(x_train.shape[1])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, nb_epoch=10,
              verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
