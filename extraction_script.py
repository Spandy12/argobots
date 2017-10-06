import urllib.request as urq
from numpy import mean

url = 'http://www.indiawaterportal.org/met_data/data/csv/22/5/5/2000/2002'
urq.urlretrieve(url, 'my_csv.csv')

# state_no = 33
# district_no = 12
# data_type = 5
# from_year = 1997
# to_year = 2002

list_of_lists = []

for v in open('my_csv.csv', 'r').read().strip().split("\n")[1:]:
    values_list = []
    for token in v.split("\t")[1:]:
        values_list.append(float(token.replace("\"", "")))
    list_of_lists.append(values_list)

# print(list_of_lists)
for l in list_of_lists:
    print(mean(l))

# kharif = []
# for li in list_of_lists[1:]:
#     kharif.append(mean(li[6:10]))

# summer = []
# for li in list_of_lists[1:]:
#     summer.append(mean(li[2:5]))
# print("For summer :- ")
# print(summer)

# winter = []
# for li in lis
