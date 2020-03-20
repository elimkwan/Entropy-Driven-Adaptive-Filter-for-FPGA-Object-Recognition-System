import pandas as pd
import numpy as np

setting = [1,2,3,4]
win_config = ["-1-24", "-4-24", "-8-24","-16-24","-23-24", "-4-12", "-8-12"]
base = True

for i in win_config:
    total = 0
    length = 0

    for j in setting:

        if base:
            filedir = "../complete-data/setting" + str(j) + i + "-base.csv"
        else:
            filedir = "../complete-data/setting" + str(j) + i + ".csv"
        

        df = pd.read_csv(filedir, skiprows=1)
        # print(df.columns)

        saved_data = df[' window_filter_time(us)']
        data = [x for x in saved_data if (~np.isnan(x) and x > 10)]

        total += np.nansum(data)
        length += len(data)
        # print("setting" + str(j))
        # print(np.nansum(data))
        # print(len(data))
        
    result = total/length
    print()
    print(i)
    print(result)
    print()


    