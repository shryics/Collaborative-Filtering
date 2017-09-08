
import os
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("ml-100k/ml-100k")
data = pd.read_csv('u.data', sep="\t", header= None)


npdata = np.zeros((944, 1683))
for i in range(len(data)):
    npdata[int(data[0][i])][data[1][i]] = data[2][i] #評価行列の生成

#npdata の　一行目　一列目　其々削除
npdata = np.delete(npdata, 0, 1)
npdata = np.delete(npdata, 0, 0)

#npdata = 評価行列, user = ユーザ行列, item = アイテム行列
k = 2 #ユーザ、アイテム行列の幅
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
Eui = npdata - np.dot(user, item)
c = 0

summ = []

print ("-----------------")
lam = 1000 #ラムダ　正則化項の係数
gan = 0.0000001 #ガンマ　学習率



while(c <= 1000):#とりあえず回しまくる
    c = c + 1
    total = 0
    print (c)
    for i in range(943):
        for j in range(1682):
            item[:,j] = item[:,j] + gan * (Eui[i,j] * user[i,:] - lam * item[:,j])
            user[i,:] = user[i,:] + gan * (Eui[i,j] * item[:,j] - lam * user[i,:])

    for i in range(943):
        for j in range(1682):
            #print(user[i,:], item[:,j])
            #print (rui[i,j])
            total = total + ( (rui[i,j] - np.dot(user[i,:], item[:,j]) ) + lam * ( (np.linalg.norm(item[:,j])) * (np.linalg.norm(item[:,j])) + (np.linalg.norm(user[i,:])) * (np.linalg.norm(user[i,:]))   ))

    summ.append(np.linalg.norm(total))
    print(np.linalg.norm(total))

def plot():
    plt.subplot(211)
    plt.plot(ite)
    plt.subplot(212)
    plt.plot(use)
    plt.show()


plt.plot(summ)
plt.show()


print("min     max")
print(np.min(np.dot(user,item)),np.max(np.dot(user,item)))
