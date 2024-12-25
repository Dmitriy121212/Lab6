import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = np.genfromtxt("usd_to_uah_rates.csv", delimiter=",", skip_header=1)
dates = np.arange(len(data)).reshape(-1, 1)
rates = data[:, 1]


dates = dates / np.max(dates)
rates = rates / np.max(rates)


X_train, X_test, y_train, y_test = train_test_split(dates, rates, test_size=0.2, random_state=30)


model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')


epochs_list = [10, 50, 100]
history_results = []

for epochs in epochs_list:
    print(f"\nНавчання з {epochs} епохами:")
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    history_results.append(history)


    y_pred = model.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {epochs} epochs: {mse}")


    sorted_indices = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]


    plt.figure(figsize=(10, 6))
    plt.plot(X_test_sorted * np.max(dates), y_test_sorted * np.max(rates), color="blue", label="Реальні значення", marker='o')
    plt.plot(X_test_sorted * np.max(dates), y_pred_sorted * np.max(rates), color="red", label="Прогноз", linestyle='-')
    plt.title(f"Прогноз курсу долара (епохи: {epochs})")
    plt.xlabel("День")
    plt.ylabel("Курс USD/UAH")
    plt.legend()
    plt.grid(True)
    plt.show()


plt.figure(figsize=(10, 6))
for i, epochs in enumerate(epochs_list):
    plt.plot(history_results[i].history['loss'], label=f'Епохи {epochs} (train)')
    plt.plot(history_results[i].history['val_loss'], label=f'Епохи {epochs} (val)', linestyle='--')
plt.title("Втрати під час навчання")
plt.xlabel("Епохи")
plt.ylabel("Втрати")
plt.legend()
plt.grid(True)
plt.show()
