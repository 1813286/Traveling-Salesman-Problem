import os
import numpy as np
from dynamicTsp.Dynamic import tspDynamic
import pandas as pd
import numpy as np
import time


def Tsp(instance):
    
    fileName = os.path.join( os.getcwd(), '../projet', 'tsp-maroc.xlsx' )
    path = fileName

    df = pd.read_excel(path, sheet_name=instance)

    df = df.replace(np.nan, 0)

    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    df = df.astype(int)

    matrix = df.to_numpy()

    if instance == 'instance_1' or instance == 'instance_3':
        distance_matrix = np.maximum(matrix, matrix.transpose())
    else:
        distance_matrix = matrix

    start_time = time.time()

    permutation, distance = tspDynamic(distance_matrix)

    end_time = time.time()
    
    timing = end_time - start_time
    
    return {'result': permutation, 'cost': distance, 'Mytime': timing}