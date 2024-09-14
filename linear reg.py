import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
data_url="http://lib.stat.cmu.edu/datasets/boston"
df=pd.read_csv(data_url,sep="\s+",skiprows=22,header=None)
data_part1 = df.values[::2, :]
data_part2 = df.values[1::2, :2]
data = np.concatenate((data_part1, data_part2), axis=1)
target = df.values[1::2, 2]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size =0.2)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
plt.scatter(y_test,y_pred)
plt.title("Linear Regression Model")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)
r2 = reg.score(x_test, y_test)
print("R-squared: ", r2)