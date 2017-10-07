# # Univariate Histograms
# import matplotlib.pyplot as plt
# import pandas
# names = ['crop', 'precip', 'max_temp', 'min_temp', 'gw', 'cc', 'season']
# data = pandas.read_csv('naive_bayes/train/data.csv', names=names)
# data.hist()
# plt.show()

# Violin Plot
# import seaborn as sns
# sns.violinplot(df['Age'], df['Gender']) #Variable Plot
# sns.despine()

# # Univariate Density Plots
# import matplotlib.pyplot as plt
# import pandas
# names = ['crop', 'precip', 'max_temp', 'min_temp', 'gw', 'cc', 'season']
# data = pandas.read_csv('naive_bayes/train/data.csv', names=names)
# data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# plt.show()

# Scatterplot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas.tools.plotting import scatter_matrix
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()
