import numpy as np

X=[]
Y=[]
for v in open("final_data.csv",'r').read().strip().split("\n"):
     value_list=[]
     for me in v.split(",")[:-1]:
         value_list.append(float(me))
     X.append(value_list)

value_list1=[]
for v in open("final_data.csv",'r').read().strip().split("\n"):
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
syn0=2*np.random.random((7,4))-1
syn1=2*np.random.random((4,2))-1
for iter in range(4000):
    l1_delta=0
    l2_delta=0
    for i in range(len(Y)):
         # forward propogation
         L0=X[i]
         L1=nonlin(np.dot(L0,syn0))
         L2=nonlin(np.dot(L1,syn1))
         #error check
         s=[[0,0]]
         index=int(Y[i])
         s[0][index]=1
         L2_error=L2-s
         L1_error=np.dot(L2_error,syn1.T)
         test=nonlin(np.dot(L0,syn0),True)
         ab = []                        
         for i in range(0, len(L1_error)):
              ab.append(L1_error[i]*test[i])    
         #multiply how much we missed with slope of sigmoid
         L0=np.array([L0])
         ab=np.array(ab)
         L1=np.array([L1])
         l1_delta+=np.dot(L0.T,ab)
         l2_delta+=np.dot(L1.T,L2_error)
         #update weights
    
    syn0=l1_delta/len(Y)
    syn1=l2_delta/len(Y)
    
       
l1=X
l2=nonlin(np.dot(l1,syn0))
l3=nonlin(np.dot(l2,syn1))
ans=[]
for data in l3:
  if data[0]>data[1]:
     ans.append("rice")
  elif data[1]>data[0]:
     ans.append("sugarcane")
     
print(ans)  










