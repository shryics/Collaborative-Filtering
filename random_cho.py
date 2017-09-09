
import os
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

os.chdir("ml-100k/ml-100k")
data = pd.read_csv('u.data', sep="\t", header= None)

lis  = [i for i in range(1682*943)]



print("list_num  i   j")

i = 0
rc = np.random.choice(lis, len(lis), replace=True)
while(i != 1682*943):


    a = rc[i]
    print(len(rc)-1682*943)
    c = a//1682
    b = a%1682
    print(a, c, b)
    i = i + 1

