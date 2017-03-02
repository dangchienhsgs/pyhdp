import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Merge
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


def create_model(dim_u, dim_v, dim_latent):
    from keras.layers import Merge

    left_branch = Sequential()
    left_branch.add(Dense(dim_latent, input_dim=dim_u))

    right_branch = Sequential()
    right_branch.add(Dense(dim_latent, input_dim=dim_v))

    merged = Merge([left_branch, right_branch], mode='dot')

    final_model = Sequential()
    final_model.add(merged)

    return final_model

    # final_model.add(Dense(10, activation='softmax'))
    # model_u = Sequential()
    # model_u.add(Dense(input_shape=(dim_u,), output_dim=dim_latent, activation='relu'))
    #
    # model_v = Sequential()
    # model_u.add(Dense(input_shape=(dim_v,), output_dim=dim_latent, activation='relu'))
    #
    # merge_model = Sequential()
    # merge_model.add(Merge([model_u, model_v], mode='dot'))
    # merge_model.add(Activation('sigmoid'))
    #
    # return merge_model


def generate_one_hot(dim, indice):
    x = np.zeros(dim)
    x[indice] = 1
    return x


def generator_data(mat):
    data_u = []
    data_v = []
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

                data_u.append(uv)
                data_v.append(iv)
                label.append(1 if mat[i, j] > 0 else 0)

            yield (np.array(data_u), np.array(data_v)), np.array(label)
        else:
            for j in np.random.permutation(np.append(zero, non_zero)):
                c += 1
                uv = generate_one_hot(n_users, i)
                iv = generate_one_hot(n_items, j)

                data_u.append(uv)
                data_v.append(iv)
                label.append(1 if mat[i, j] > 0 else 0)
                co += 1

                if co % 10 == 0 or co == num_samples * 2 / 10:
                    yield [np.array(data_u), np.array(data_v)], np.array(label)
                    data_u = []
                    data_v = []
                    label = []


def generator_data_test(mat):
    data_u = []
    data_v = []
    label = []

    for i in range(mat.shape[0]):
        non_zero = mat[i, :].nonzero()[0]

        if i % 100 == 0:
            print("Read test at user ", i)

        for j in non_zero:
            uv = generate_one_hot(n_users, i)
            iv = generate_one_hot(n_items, j)

            data_u.append(uv)
            data_v.append(iv)
            label.append(1)

    return (data_u, data_v), np.array(label)


# training and testing
model = create_model(n_users, n_items, 8)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit_generator(generator=generator_data(train), nb_epoch=10,
                    samples_per_epoch=18000,
                    nb_val_samples=60,
                    validation_data=generator_data(test))

score = model.evaluate_generator(generator_data(test), val_samples=900)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#
# print(element_wise(np.array([1, 2]), np.array([3, 5])))

