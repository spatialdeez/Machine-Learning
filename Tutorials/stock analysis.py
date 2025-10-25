# Using simple machine learning code to create predicitve stock analysis model
# WARNING: This is not to be used to 100% predict and buy stocks. Always check from other sources. Use the data with caution
# Using RandomForestClassifier, which trains multiple decision trees with random parameters, and lastly averaging the results of these trees.
# -> Less chance for overfitting which is when a model is too used to the training dataset and cannot predict well to the new ones
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from googlesearch import search

ticker = 'AAPL'
stock = yf.Ticker(ticker=ticker)
data = stock.history(period="10y")
query = 'recent+news+about+' + ticker


remove_columns = ['Dividends', 'Stock Splits']
data = data.drop(columns=remove_columns, axis=1)

data['Tomorrow'] = data['Close'].shift(-1) # Find the prices after the dates. e.g. the stock price is $5 at 15 december so at 14 december the tomorrow column is $5
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int) # Convert the boolean into 1 and 0. 1 is if the price went up and 0 is if the price went down


# Data calculations not written by me 
# Calculate normalized/ratio-based features
data['Open_Close_Ratio'] = data['Open'] / data['Close']
data['High_Low_Ratio'] = data['High'] / data['Low']
data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
data['Daily_Return'] = data['Close'].pct_change()

# Moving averages for trend context
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_20'] = data['Close'].rolling(20).mean()
data['MA_Ratio'] = data['MA_5'] / data['MA_20']

# Volume-based
data['Volume_MA'] = data['Volume'].rolling(10).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

# Remove NaN values created by rolling calculations and percentage changes
data = data.dropna()

predictors = ['Open_Close_Ratio', 'High_Low_Ratio', 'Price_Range', 
              'Daily_Return', 'MA_Ratio', 'Volume_Ratio']

# start the model
'''
n_estimators is the amount of decision trees
min_samples_split is used to reduce overfitting. also controls the minimum number of samples required to split an internal node in any of the individual decision trees within the forest.
random_state is the random seed. Good for testing if it is set to a constant such as 1
'''
model = RandomForestClassifier(n_estimators=500, min_samples_split=20, random_state=1) 
train = data.iloc[:-100] # give some data to train, Except for the last 100 rows of data
test = data.iloc[-100:] # give some data to test. Except for the first 100 rows of data because its used for training

model.fit(train[predictors], train['Target']) # fit data into the model for training
predictions = model.predict(test[predictors]) # let the model try and predict
predictions = pd.Series(predictions, index=test.index)
accuracy = precision_score(test['Target'], predictions)
print('current model accuracy:' + str(accuracy) + '%')


# Try and predict next day data
# Get the most recent data (today's data)
latest_data = data.iloc[-1:].copy()
today_features = latest_data[predictors]

# Try and make prediction for tomorrow
tomorrow_prediction = model.predict(today_features)[0]
tomorrow_prediction_proba = model.predict_proba(today_features)[0]

# Contextual data for the model to use
current_price = latest_data['Close'].iloc[0]
today_open = latest_data['Open'].iloc[0]
today_high = latest_data['High'].iloc[0]
today_low = latest_data['Low'].iloc[0]

print(f"Current Price: ${current_price:.2f}")
print(f"Today's Range: ${today_low:.2f} - ${today_high:.2f}")
print(f"Prediction for Tomorrow: {'UP' if tomorrow_prediction == 1 else 'DOWN'}")
print(f"Confidence: {tomorrow_prediction_proba[tomorrow_prediction]:.2%}")
print(f"Probability Breakdown:")
print(f"  - UP: {tomorrow_prediction_proba[1]:.2%}")
print(f"  - DOWN: {tomorrow_prediction_proba[0]:.2%}")
print('DISCLAIMER: Use the data above with caution. Always check your sources.')
print('Useful link(s) related to your preferred stock:')
print(f"https://www.google.com/search?q={query}&tbm=nws")



# test zone
combined = pd.concat([test["Target"], predictions], axis=1) # Create a new table combining the real data with the preditions and create columns
combined.plot() # plot table on graph
# plot the line
# data['Close'].plot.line()
# plt.show()