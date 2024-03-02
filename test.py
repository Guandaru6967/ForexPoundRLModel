import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import datetime
import mplfinance as mplf
from tapy.indicators import Indicators

import multiprocessing

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)

