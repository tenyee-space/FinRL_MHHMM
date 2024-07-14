from scipy.io import arff
import pandas as pd 

file_name='/root/autodl-tmp/SharePriceIncrease/SharePriceIncrease_TRAIN.csv'

# data,meta=arff.loadarff(file_name)
# #print(data)
# print(meta)

# df=pd.DataFrame(data)
# print(df.head())
# #print(df)

# #保存为csv文件
# out_file='/root/FinRL_DCHMM/DCHMM/FingerMovements_TEST.csv'
# output=pd.DataFrame(df)
# output.to_csv(out_file,index=False)
df = pd.read_csv(file_name)

print(df.head())
