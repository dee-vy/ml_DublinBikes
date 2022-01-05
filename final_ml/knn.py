import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# read data. column 1 is date/time,col 3 is the station name and col 6 is #bikes
x = pd.read_csv("../dublinbikes_20200101_20200401.csv", usecols=[1, 3, 6], parse_dates=[1])

df = x[x['NAME'] == "GRANGEGORMAN LOWER (SOUTH)"]
# PORTOBELLO HARBOUR, DAME STREET, PEARSE STREET, UPPER SHERRARD STREET.
# Most busy:HANOVER QUAY  Least busy: GRANGEGORMAN LOWER (SOUTH)

print(df)
print("Dataset size : ", np.shape(df))
# 31/1/20 (Friday) to 12/3/20 (Thursday). 6 weeks data.
start = pd.to_datetime("31-01-2020", format='%d-%m-%Y')
end = pd.to_datetime("12-03-2020", format='%d-%m-%Y')
print("Start date : ", start)
# convert date/time to unix timestamp in sec
t_full = pd.array(pd.DatetimeIndex(df.iloc[:, 0]).view(np.int64)) / 1000000000
# print("Time : ", np.shape(t_full))
# print("Time : ", np.shape(df.iloc[:, 2]))
plt.scatter((t_full - t_full[0]) / (24 * 3600), df.iloc[:, 2])
plt.xlabel("time (days)")
plt.ylabel("#bikes")
plt.legend(["Bikes"], loc='upper right')
plt.show()

# extract data between start and end dates
t_start = pd.DatetimeIndex([start]).view(np.int64) / 1000000000
t_end = pd.DatetimeIndex([end]).view(np.int64) / 1000000000
temp = np.extract([(t_full >= t_start[0])], t_full)
t = np.extract([(temp <= t_end[0])], temp)
t = (t - t[0]) / 60 / 60 / 24  # convert timestamp to days
y = np.extract(temp, df.iloc[:, 2])  # [(t_full >= t_start[0])]
y = np.extract([(temp <= t_end[0])], y).view(np.int64)
plt.scatter(t, y)
plt.xlabel("time (days)")
plt.ylabel("#bikes")
plt.legend(["Bikes"], loc='upper right')
plt.show()

dt = t_full[1] - t_full[0]
print("Data sampling interval is %d secs (%d minutes)" % (dt, dt / 60))

# Time= 10 , 30  and 60 minutes. Hence, q = 2, 6 and 12. (time=q*5)
q = 2
lag = 3
stride = 1
w = math.floor(7 * 24 * 60 * 60 / dt)  # number of samples per week
length = y.size - lag * w - q
XX = y[q:q + length:stride]
# print("XX : ", XX)

for i in range(1, lag):  # last three weeks' data
    X = y[i * w + q:i * w + q + length:stride]
    XX = np.column_stack((XX, X))

d = math.floor(24 * 60 * 60 / dt)  # number of samples per day
for i in range(1, lag + 1):  # last three days' data
    X = y[lag * w - i * d + q:lag * w - i * d + q + length:stride]
    XX = np.column_stack((XX, X))

for i in range(0, lag + 3, 2):  # To take 10 min interval instead of 3 for short-term trends in data.
    X = y[lag * w - i:lag * w - i + length:stride]
    XX = np.column_stack((XX, X))

yy = y[lag * w + q:lag * w + q + length:stride]
tt = t[lag * w + q:lag * w + q + length:stride]

print("Choosing parameter, k, for kNN Regressor using Cross-validation.")
mean_error = []
std_error = []
temp = []
# k_range = [70, 100, 150, 200, 250, 275, 280, 285, 290, 300, 320, 350, 400, 500, 600]
# k_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for k in k_range:
    print("k:", k)
    kf = KFold(n_splits=5)
    for train, test in kf.split(XX):
        model = KNeighborsRegressor(n_neighbors=k).fit(XX[train], yy[train])
        ypred = model.predict(XX[test])
        # r2 = r2_score(y_pred=ypred, y_true=yy[test])
        mse = mean_squared_error(yy[test], ypred)
        temp.append(mse)
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    print("Mean Squared Error :", np.array(temp).mean())

plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('k')
plt.ylabel('Mean Squared Error')
plt.show()

train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
# train = np.arange(0,yy.size)

model = KNeighborsRegressor(n_neighbors=9).fit(XX[train], yy[train])
ypred = model.predict(XX[test])
print("Model score : ", r2_score(y_pred=ypred, y_true=yy[test]))

# Compare against a Baseline
yy_baseline = XX[test][:, 6]  # Using current value as next prediction.
# (Baseline is assuming that next value is similar to the current value)
print("Baseline score : ", r2_score(y_true=yy[test], y_pred=yy_baseline))

if True:
    y_pred = model.predict(XX)
    plt.scatter(tt, yy, color='black')
    plt.scatter(tt, y_pred, color='blue')
    plt.xlabel("time (days)")
    plt.ylabel("#bikes")
    plt.legend(["training data", "predictions"], loc='upper right')
    # day = math.floor(24 * 60 * 60 / dt)  # number of samples per day
    # plt.xlim((4 * 7, 4 * 7 + 4))
    plt.show()
