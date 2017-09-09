
import os
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
        user[i][j] = random.randint(0,1)
###item 乱数による初期化
for i in range(1682):
    for j in range(k):
        item[j][i] = random.randint(0,1)

rui = npdata
Eui = rui - np.dot(user, item)
c = 0
summ = []


lam = 0.001 #ラムダ　正則化項の係数
gan = 0.01  #ガンマ　学習率

lis  = [i for i in range(1682*943)]


Eui = 0
while(c <= 5000):
    c = c + 1
    rc = np.random.choice(lis, len(lis), replace=True)


#####ユーザ行列とアイテム行列の更新#####
    l = 0
    while (l != 1682 * 943):
        a = rc[l]
        s = a // 1682 # i
        t = a % 1682 # j
        l = l + 1
        if rui[s][t] > 0:
            Eui = rui[s][t] - np.dot(user[s, :], item[:, t])
            item[:, t] = item[:, t] + (gan / math.sqrt(c + 1)) * (Eui * user[s, :] - lam * item[:, t])
            user[s, :] = user[s, :] + (gan / math.sqrt(c + 1)) * (Eui * item[:, t] - lam * user[s, :])


    if c == 1:
        err_b = -1

#####誤差の算出#####
    err = 0
    for i in range(943):
        for j in range(1682):
            if rui[i][j] > 0:
                err = err + ( (rui[i, j] - np.dot(user[i, :], item[:, j]) ) * (rui[i, j] - np.dot(user[i, :], item[:, j]) ) + lam * (
                (np.linalg.norm(item[:, j])) * (np.linalg.norm(item[:, j])) + (np.linalg.norm(user[i, :])) * (
                np.linalg.norm(user[i, :]))))



    summ.append(err/100000)
    print(err/100000, abs(err/100000 - err_b), c)

#####前との誤差の差が 0.000001より小さければ終了
    if abs(err/100000 - err_b) < 0.001:
        break
    err_b = err/100000

plt.plot(summ)
plt.show()


print("min     max")
print(np.min(np.dot(user,item)),np.max(np.dot(user,item)))
print("min  u   max")
print(np.min(user),np.max(user))
print("min  i   max")
print(np.min(item),np.max(item))

print(item)
print(user.T)

