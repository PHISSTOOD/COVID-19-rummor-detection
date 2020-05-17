import os
import pandas as pd
import numpy as np


inputfile_dir1 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Tbath_raw.csv"
inputfile_dir2 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\sip_raw.csv"
inputfile_dir3 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Salt_raw.csv"
inputfile_dir4 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\mosquito_raw.csv"
inputfile_dir5 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\Cold_raw.csv"
inputfile_dir6 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\garlic_raw.csv"
inputfile_dir7 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\hold_raw.csv"
inputfile_dir8 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\transmitted_raw.csv"
"""inputfile_dir9 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\salt water.csv"
inputfile_dir10 = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\sipping water.csv"
"""
outputfile = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\all_tweets3.csv"
if __name__ == '__main__':
        df1 = pd.read_csv(inputfile_dir1, header=None)
        df2 = pd.read_csv(inputfile_dir2, header=None, skiprows=1)
        df3 = pd.read_csv(inputfile_dir3, header=None, skiprows=1)
        df4 = pd.read_csv(inputfile_dir4, header=None, skiprows=1)
        df5 = pd.read_csv(inputfile_dir5, header=None, skiprows=1)
        df6 = pd.read_csv(inputfile_dir6, header=None, skiprows=1)
        df7 = pd.read_csv(inputfile_dir7, header=None, skiprows=1)
        df8 = pd.read_csv(inputfile_dir8, header=None, skiprows=1)
        #df9 = pd.read_csv(inputfile_dir9, header=None, skiprows=1)
        #df10 = pd.read_csv(inputfile_dir10, header=None, skiprows=1)
        """frames = [df1, df2, df3,df4,df5,df6,df7,df7,df9,df10]
        result = pd.concat(frames,ignore_index=True)"""
        df = pd.DataFrame(np.concatenate([df1.values, df2.values, df3.values,df4.values, df5.values, df6.values,df7.values, df8.values]), columns = df1.columns)
        df.to_csv(outputfile, mode='a', index=False, header=False)
