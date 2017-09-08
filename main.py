
import os
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("ml-100k/ml-100k")
data = pd.read_csv('u.data', sep="\t", header= None)
print (data.shape)
print (max(data[0]))
print (max(data[1]))

npdata = np.zeros((944, 1683))

for i in range(len(data)):
    npdata[int(data[0][i])][data[1][i]] = data[2][i] #評価行列の生成
#npdata の　一行目　一列目　其々削除
print (type(npdata))
npdata = np.delete(npdata, 0, 1)
npdata = np.delete(npdata, 0, 0)

#npdata = 評価行列, user = ユーザ行列, item = アイテム行列
k = 2 #ユーザ、アイテム行列の幅
user = np.zeros((943, k)) #zerosにする！！！
item = np.zeros((k, 1682))

###user 乱数による初期化
for i in range(943):
    for j in  range(k):
        user[i][j] = random.randint(1,5)
###item 乱数による初期化
for i in range(1682):
    for j in range(k):
        item[j][i] = random.randint(1,5)




Eui = npdata - np.dot(user, item)
#print (Eui)

c = 0
ite = []
use = []
summ = []

print ("-----------------")
lam = 1000 #ラムダ　正則化項の係数
gan = 0.0000001 #ガンマ　学習率

testes = Eui

while(c <= 2000):#とりあえず回しまくる
    c = c + 1

    item = item + gan * ( (np.dot(Eui.T, user)).T - lam * item)
    user = user + gan * ( np.dot(Eui, item.T) - lam * user)

    ite.append(np.linalg.norm(item))
    use.append(np.linalg.norm(user))
    #if (np.linalg.norm(testes) > (np.linalg.norm(Eui - np.dot(user,item)))):
    #    break
    testes = Eui - np.dot(user,item)

    #sum = (Eui - np.dot(user,item))^2 + 1000 * ( (np.linalg.norm(item))*(np.linalg.norm(item)) + (np.linalg.norm(user))*(np.linalg.norm(item)) )
    #for i in range(943):
    #    for j in range(1682):
     #       sum1 = sum1 + (testes[i][j]) * (testes[i][j])
      #      sum2 = sum2 + lam * ( (np.linalg.norm(user[i,:])) * (np.linalg.norm(user[i,:])) + (np.linalg.norm(item[:,j])) * (np.linalg.norm(item[:,j])) )



    summ.append(np.linalg.norm(testes))
    print(np.linalg.norm(testes))

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
