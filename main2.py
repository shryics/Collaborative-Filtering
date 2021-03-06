
import os
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

os.chdir("ml-100k/ml-100k")
data = pd.read_csv('u.data', sep="\t", header= None)


npdata = np.zeros((944, 1683))
for i in range(len(data)):
    npdata[int(data[0][i])][data[1][i]] = data[2][i] #評価行列の生成

#npdata の　一行目　一列目　其々削除
npdata = np.delete(npdata, 0, 1)
npdata = np.delete(npdata, 0, 0)

#npdata = 評価行列, user = ユーザ行列, item = アイテム行列
k = 30 #ユーザ、アイテム行列の幅
user = np.zeros((943, k))
item = np.zeros((k, 1682))

###user 乱数による初期化
for i in range(943):
    for j in  range(k):
        user[i][j] = random.randint(1,5)
###item 乱数による初期化
for i in range(1682):
    for j in range(k):
        item[j][i] = random.randint(1,5)

rui = npdata
Eui = rui - np.dot(user, item)
c = 0

summ = []

print ("-----------------")
lam = 0.1 #ラムダ　正則化項の係数
gan = 0.001
#ガンマ　学習率


Eui = 0
while(c <= 100):
    c = c + 1
    total = 0
    for i in range(943):
        for j in range(1682):
            if rui[i][j] > 0:
                Eui = rui[i][j] - np.dot(user[i,:], item[:,j])
                #ここの二行をkにするといいのでは？
                #
                item[:,j] = item[:,j] + (gan/math.sqrt(c+1)) * (Eui * user[i,:] - lam * item[:,j])
                user[i,:] = user[i,:] + (gan/math.sqrt(c+1)) * (Eui * item[:,j] - lam * user[i,:])


    err = 0
    for i in range(943):
        for j in range(1682):
            if rui[i][j] > 0:
                err = err + ( (rui[i, j] - np.dot(user[i, :], item[:, j]) ) * (rui[i, j] - np.dot(user[i, :], item[:, j]) ) + lam * (
                (np.linalg.norm(item[:, j])) * (np.linalg.norm(item[:, j])) + (np.linalg.norm(user[i, :])) * (
                np.linalg.norm(user[i, :]))))

    #Eui = rui - np.dot(user, item)

    summ.append(err/100000)
    print(err/100000, c)

plt.plot(summ)
plt.show()


print("min     max")
print(np.min(np.dot(user,item)),np.max(np.dot(user,item)))



print(item)

print(user.T)

