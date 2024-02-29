from sklearn.preprocessing import  MinMaxScaler,MultiLabelBinarizer,Normalizer
import numpy as np
import pandas as pd
from typing import Tuple
# Assuming df is your 5-minute price dataset with a 'price' column
# Replace 'price' with the actual column name in your dataset

# Extract the feature to be normalized
class PriceImagefier:
        def __init__(self,price_data:pd.DataFrame,shape=[512,512]) -> None:
                self.normalizer=scaler = MinMaxScaler()
                self.price_data=price_data
                all_prices=price_data[["Open","High","Close","Low"]]
                self.normalized_prices = scaler.fit_transform(all_prices)
                print("Before Normalization\n:",all_prices.head(10))
                print("After Normalization\n:",self.normalized_prices.head(10))

        def process(self,price_dataframe,save=True)->Tuple[np.ndarray]:
               
                price_dataframe.reset_index(inplace=True)
                
                scaler=MinMaxScaler()
                data=pd.DataFrame(scaler.fit_transform(price_dataframe), columns=price_dataframe.columns, index=price_dataframe.index)
                print(data)


                
                



# Initialize the MinMaxScaler

