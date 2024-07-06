from scipy.io import arff
import pandas as pd 

file_name='/root/autodl-tmp/FingerMovements_TEST.arff'

data,meta=arff.loadarff(file_name)
#print(data)
print(meta)

df=pd.DataFrame(data)
print(df.head())
#print(df)

#保存为csv文件
# out_file='/root/autodl-tmp/FingerMovements_TEST.csv'
# output=pd.DataFrame(df)
# output.to_csv(out_file,index=False)
