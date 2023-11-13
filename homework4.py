from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# нормализует данные в датафрейме
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def predict(k, test_X):
    neighbors = []
    for x in test_X:
        distances = np.sqrt(np.sum((x - X_train) ** 2, axis=1))
        y_sorted = [y for _, y in sorted(zip(distances, y_train))]
        neighbors.append(y_sorted[:k])
    return list(map(lambda x: int(Counter(x).most_common(1)[0][0]), neighbors))


def plot_projections(X, y, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for (i, j), ax in zip(combinations, axs.ravel()):
        ax.scatter(X.iloc[:, i], X.iloc[:, j], c=y, cmap=plt.cm.jet)
        ax.set_xlabel(iris_df.columns[i])
        ax.set_ylabel(iris_df.columns[j])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# датасет
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Разделение на обучающую и тестовую выборку
# случайным образом разделяет данные

X = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

data_normalized = normalize(X)

X_train = X.sample(frac=0.8, random_state=0)
y_train = y.sample(frac=0.8, random_state=0)

X_test = X.drop(X_train.index)
y_test = y.drop(y_train.index)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# определяем оптимальное значение k
scores = []
for k in range(1, 50):
    correct_predictions = 0
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        k_nearest_neighbors = y_train[sorted_indices[:k]]

        prediction = np.argmax(np.bincount(k_nearest_neighbors.astype(int)))
        if prediction == y_test[i]:
            correct_predictions += 1
    scores.append(correct_predictions / len(X_test))

optimal_k = np.argmax(scores) + 1
print(np.argmax(scores))
print(f"Оптимальное значение k: {optimal_k}")

plot_projections(X, y, "Проекции до нормализации")
plot_projections(data_normalized, y, "Проекции после нормализации")

data_dict = {}
column_names = X.columns  # Получите имена колонок из исходного DataFrame X

for name in column_names:
    value = float(input(f"Введите значение для {name}: "))
    data_dict[name] = [value]

# Создайте новый DataFrame на основе словаря
new_sample_df = pd.DataFrame(data_dict)
# input


new_sample_normalized = normalize(new_sample_df)
new_sample_normalized = np.asarray(new_sample_normalized)
prediction = predict(optimal_k, new_sample_normalized)

print(f"Класс нового объекта: {iris.target_names[prediction[0]]}")
