import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



class FinanceNews:
    def __init__(self, path):
        self.path = path
        
        self.train, self.test = self.load()

    def load(self):
        df = pd.read_csv(self.path)
        def t2l(txt):
            if txt=='positive':
                return 2.0
            if txt=='negative':
                return 0.0
            if txt=='neutral':
                return 1.0
            raise ValueError(f"expected positive/negative/neutral, git {txt}")

        df['label'] = df['label'].apply(lambda x : t2l(x))
        train, test = train_test_split(df,train_size=0.7,test_size=0.3,random_state=42)

        return train, test
    

    
class TwitterNews:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train, self.test = self.load()

    def load(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        return train, test



def merge_df(df1, df2):
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    concat_df = pd.concat((df1,df2),ignore_index=True)
    return concat_df
