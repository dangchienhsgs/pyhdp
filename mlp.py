import numpy as np
import pandas as pd
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/grouplens/u.data', sep='\t', names=names)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print(str(n_users) + ' users')
print(str(n_items) + ' items')

ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()

    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


train, test = train_test_split(ratings)
print("Train shape {0}".format(train.shape))
print("Test shape {0}".format(test.shape))
num_samples = len(train.reshape(n_users * n_items, ).nonzero()[0])
print("Num samples of train ", num_samples)


def create_model(dim):
    model = Sequential()
    model.add(Dense(input_shape=(dim,), output_dim=32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def create_model_gmf():
    model = Sequential()
    return model


def element_wise(a, b):
    return np.array(np.array([[a]]).transpose(), np.array([b]))


def generate_one_hot(dim, indice):
    x = np.zeros(dim)
    x[indice] = 1
    return x


def generator_data(mat):
    data = []
    label = []

    co = 0
    c = 0
    print(mat.shape)
    print(len(mat[942, :].nonzero()[0]))
    print(len(mat[942, :].nonzero()[0]))
    for i in range(mat.shape[0]):
        non_zero = np.array(mat[i, :]).nonzero()[0]
        zero = np.random.choice(np.where(np.array(mat[i, :]) == 0)[0], size=len(non_zero), replace=False)

        if i == n_users - 1:
            print(c, num_samples * 2 - c, len(zero) + len(non_zero))
            for j in np.random.permutation(np.append(zero, non_zero)):
                uv = generate_one_hot(n_users, i)
                iv = generate_one_hot(n_items, j)

                data.append(np.concatenate([uv, iv]))
                label.append(1 if mat[i, j] > 0 else 0)

            yield np.array(data), np.array(label)
        else:
            for j in np.random.permutation(np.append(zero, non_zero)):
                c += 1
                uv = generate_one_hot(n_users, i)
                iv = generate_one_hot(n_items, j)

                data.append(np.concatenate([uv, iv]))
                label.append(1 if mat[i, j] > 0 else 0)
                co += 1

                if co % 60 == 0 or co == num_samples * 2 / 10:
                    yield np.array(data), np.array(label)
                    data = []
                    label = []


def generator_data_test(mat):
    data = []
    label = []

    for i in range(mat.shape[0]):
        non_zero = mat[i, :].nonzero()[0]

        if i % 100 == 0:
            print("Read test at user ", i)

        for j in non_zero:
            uv = generate_one_hot(n_users, i)
            iv = generate_one_hot(n_items, j)
            data.append(np.concatenate([uv, iv]))
            label.append(1)

    return np.array(data), np.array(label)


# training and testing
model = create_model(n_users + n_items)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

test_data = generator_data_test(test)
model.fit_generator(generator=generator_data(train), nb_epoch=10,
                    samples_per_epoch=18000,
                    validation_data=test_data)

score = model.evaluate(test_data[0], test_data[1])
print('Test score:', score[0])
print('Test accuracy:', score[1])
