# Univariate Histograms
import matplotlib.pyplot as plt
import pandas
names = ['crop', 'precip', 'max_temp', 'min_temp', 'gw', 'cc', 'season']
data = pandas.read_csv('naive_bayes/train/data.csv', names=names)
data.hist()
plt.show()
