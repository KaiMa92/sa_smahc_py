# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:56:54 2023

@author: kaiser
"""
from datetime import datetime
import os
import json

class file_creator:
    def __init__(self, dir_path,params_dct, data_df):
        file_name = 'data_output_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
        path = os.path.join(dir_path,file_name)
        with open(path, 'w') as file:
            params_string = json.dumps(params_dct) + '\n'
            file.write(params_string)
        data_df.to_csv(path, sep = '\t', mode = 'a', index = False)   