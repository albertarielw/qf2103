import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

def extract_train_and_test_data(data):
    df = data.dropna()
    df.reset_index(drop=True, inplace=True)
    even_rows = df.iloc[df.index % 2 == 0]
    odd_rows = df.iloc[df.index % 2 != 0]
    return odd_rows.dropna().to_numpy(), even_rows.dropna().to_numpy()

def predict_position(train, test):
    # Train Hidden Markov Model
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
    model.fit(train.reshape(-1, 1))

    # Predict hidden states for test data
    hidden_states = model.predict(test.reshape(-1, 1))

    # Define a function to map hidden states to rise or fall
    def map_to_rise_or_fall(state):
        return "Rise" if state == 0 else "Fall"

    # Map hidden states to rise or fall
    predicted_rise_fall = np.array(list(map(map_to_rise_or_fall, hidden_states)))

    return predicted_rise_fall

def trade(balance, test, prediction):
    position = 0

    for today in range(len(test) - 1):
        tomorrow = today + 1
        if prediction[tomorrow] == "Rise":
            balance -= (1 - position) * test[today]
        else:
            balance += (position + 1) * test[today]
    
    balance += position * test[-1]

    return balance

def backtest(data):
    train_data, test_data = extract_train_and_test_data(data)

    N = 1000
    profit = 0
    for _ in range(N):
        prediction = predict_position(train_data, test_data)
        profit += trade(0, test_data, prediction)

    return profit / N

def run_simulation():
    raw = pd.read_csv('tr_eikon_eod_data.csv', index_col = 0, parse_dates = True)
    stock_names = ['AAPL.O', 'MSFT.O', 'INTC.O', 'AMZN.O', 'GS.N']
    # stock_names = ['AMZN.O']
    for stock_name in stock_names:
        print(f"{stock_name}: ${backtest(raw[stock_name])}")


def run_simulation_2():
    stock_names = ['AAPL.O', 'MSFT.O', 'INTC.O', 'AMZN.O', 'GS.N']
    raw = pd.read_csv('tr_eikon_eod_data.csv', index_col = 0, parse_dates = True)
    raw = raw[stock_names]
    raw = raw.dropna()

    all_predictions = {}
    all_train_data = {}
    all_test_data = {}
    N = 1000

    for stock_name in stock_names:
        train_data, test_data = extract_train_and_test_data(raw[stock_name])
        all_train_data[stock_name] = train_data
        all_test_data[stock_name] = test_data
        all_predictions[stock_name] = []
        for _ in range(N):
            all_predictions[stock_name].append(predict_position(train_data, test_data))
    
    def calculate_profit(test, prediction):
        balance = 0
        position = 0

        for today in range(len(test) - 1):
            tomorrow = today + 1
            if prediction[tomorrow] == "Rise":
                balance -= (1 - position) * test[today]
            else:
                balance += (position + 1) * test[today]
        
        balance += position * test[-1]

        return balance
    
    total_profit = 0
    for stock_name in stock_names:
        print(stock_name)
        best_profit = -float('inf')
        for other_stock_name in stock_names:
            profit = 0
            for prediction in all_predictions[other_stock_name]:
                profit += calculate_profit(all_test_data[stock_name], prediction)
            profit = profit / N
            print(f"{other_stock_name} {profit}")
            best_profit = max(profit, best_profit)
        print(best_profit)
        total_profit += best_profit
    print(f"total: {total_profit}")

run_simulation_2()