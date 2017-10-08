import numpy as np
import matplotlib.pyplot as plt
import get_data as gd
X=[]

X = gd.get_features()

X=np.array(X)

f1_rice=[]
f2_rice=[]
for i in range(0,500):
        f1_rice.append(X[i][5])
        f2_rice.append(X[i][3])

f1_sugar=[]
f2_sugar=[]
for j in range(500,900):
        f1_sugar.append(X[j][5])
        f2_sugar.append(X[j][3])

f1_maize=[]
f2_maize=[]
for j in range(900,1298):
        f1_maize.append(X[j][5])
        f2_maize.append(X[j][3])
#plt.plot(f1_rice,'ro',f1_maize,'bs',f1_sugar,'g^',alpha=0.25)
plt.plot(f1_rice,f2_rice,'ro',f1_maize,f2_maize,'bs',f1_sugar,f2_sugar,'g^',alpha=0.25)
plt.show()
