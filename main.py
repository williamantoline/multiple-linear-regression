import numpy as np
from scipy.linalg import inv
import pandas as pd
import math


# calc beta(s)
def calc_beta(X, Y):
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    X_transpose = np.transpose(X_with_intercept)
    X_transpose_X = np.dot(X_transpose, X_with_intercept)
    X_transpose_X_inv = inv(X_transpose_X)
    X_transpose_X_inv_X_transpose = np.dot(X_transpose_X_inv, X_transpose)

    beta = np.dot(X_transpose_X_inv_X_transpose, Y)
    return beta

# calc predicted value (y')
def calc_pred(row, b0, b1, b2, b3):
    return b0 + b1 * row['x1'] + b2 * row['x2'] + b3 * row['x3'] 

# evaluate correlation relationship (negative or positive)
def evaluate_corr_rels(corr):
    if corr == 0:
        rels = "none"
    elif corr > 0:
        rels = "positive"
    else:
        rels = "negative"
    return rels
    
# evaluate correlation strength
def evaluate_corr_type(corr):
    abscorr = abs(corr)
    if abscorr >= 0.8:
        typ = "strong"
    elif abscorr >= 0.6:
        typ = "moderate"
    elif abscorr >= 0.4:
        typ = "weak"
    elif abscorr > 0:
        typ = "very weak"
    else:
        typ = "none"
    return typ

def get_strongest_corr(a,b,c):
    mx = max(a,max(b,c))
    if mx == a:
        return "x1"
    elif mx == b:
        return "x2"
    else:
        return "x3"


# pandas configuration
pd.set_option('display.max_rows', None)

# Read csv data
df = pd.read_csv("data.csv")

# define x and y
X = df[["x1","x2","x3"]]
Y = df["y"]

# calculate coeffs b0,b1,b2,b3
[b0,b1,b2,b3] = calc_beta(X, Y)

# append helping values
df["y'"] = df.apply(calc_pred, axis=1, args=(b0, b1, b2, b3))
df["y-y'"] = df["y"] - df["y'"]
df["|y-y'|"] = abs(df["y-y'"])
df["|y-y'|^2"] = pow(df["y-y'"], 2)

# append a and b to calculate correlations
df["a1"] = df["y"].mean() - df["x1"]
df["a2"] = df["x1"].mean() - df["x2"]
df["a3"] = df["x2"].mean() - df["x3"]
df["b"] = df["x3"].mean() - df["y"]

# calculate MAE, MSE, RMSE
mae = df["|y-y'|"].mean()
mse = df["|y-y'|^2"].mean()
rmse = math.sqrt(mse)

# calculate Correlation of x1, x2, x3
corr_a1 = df["a1"].corr(df["b"])
corr_a2 = df["a2"].corr(df["b"])
corr_a3 = df["a3"].corr(df["b"])


print()
print(df)
print()
print("MAE:\t\t", mae)
print("MSE:\t\t", mse)
print("RMSE:\t\t", rmse)
print("Corr X1:\t", corr_a1, evaluate_corr_type(corr_a1), evaluate_corr_rels(corr_a1))
print("Corr X2:\t", corr_a2, evaluate_corr_type(corr_a2), evaluate_corr_rels(corr_a2))
print("Corr X3:\t", corr_a3, evaluate_corr_type(corr_a3), evaluate_corr_rels(corr_a3))
print("Strongest Corr: ", get_strongest_corr(corr_a1,corr_a2,corr_a3))
