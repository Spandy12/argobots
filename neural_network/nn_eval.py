import numpy as np
from sklearn.metrics import accuracy_score

X=[]
Y=[]
for v in open("data4.csv",'r').read().strip().split("\n"):
     value_list=[]
     for me in v.split(",")[:-1]:
         value_list.append(float(me))
     X.append(value_list)

value_list1=[]
for v in open("data4.csv",'r').read().strip().split("\n"):
     value_list1.append(float(v[-1]))
Y.append(value_list1)

X=np.array(X)
Y=np.array(Y).T

def nonlin(x,deriv=False):
    if deriv==True:
      return nonlin(x)*nonlin(1-x)
    return 1/(1+np.exp(-x))

np.random.seed(1)

#weights
syn0=2*np.random.random((7,2))-1
syn1=2*np.random.random((4,2))-1
for iter in range(3000):
    for i in range(len(Y)):
         # forward propogation
         L0=X[i]
         L1=nonlin(np.dot(L0,syn0))
         #error check
         s=[[0,0]]
         index=int(Y[i])
         s[0][index]=1
         L1_error=L1-s

         #multiply how much we missed with slope of sigmoid
         l1_delta=L1_error*nonlin(L1,True)
         #update weights
         L0=np.array([L0])
         syn0 +=np.dot(L0.T,l1_delta)

L0=X
L2=nonlin(np.dot(L0,syn0))
print("output")
print(L2)
ans=[]
for data in L2:
     if data[0]>=data[1]:
          ans.append(0)
     elif data[0]<data[1]:
          ans.append(1)


print(accuracy_score(Y, np.array(ans)))
