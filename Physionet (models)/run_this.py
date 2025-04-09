import pandas as pd
from run_experiments import run_experiments
import random
import os
import sys
import numpy as np
import tensorflow as tf


# zeros=[1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 37, 39, 40, 41, 43, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 63, 64, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 85, 86, 87, 90, 91, 93, 94, 96, 97, 99, 101, 102, 105, 106, 107, 109]

# filenames = [f"S{str(e).zfill(3)}" for e in zeros if len(f"S{str(e).zfill(3)}") == 4]
# full_res=[]

#runs=5

# for run in range(runs):
#     subs_considered = random.sample(filenames, 6)+["S018"]
#     temp_res=run_experiments(subs_considered,full_res)
#     full_res.append([subs_considered,temp_res])
# subs_considered = ["S007","S009","S010","S011","S012","S013 ","S018"]
# temp_res=run_experiments(subs_considered,full_res)
# full_res.append([subs_considered,temp_res])
# full_res.append("##################")


# # for run in range(runs):
# #     subs_considered = random.sample(filenames, 6)+["S011"]
# #     temp_res=run_experiments(subs_considered,full_res)
# #     full_res.append([subs_considered,temp_res])
# subs_considered = ["S007","S009","S010","S018","S012","S013 ","S011"]
# temp_res=run_experiments(subs_considered,full_res)
# full_res.append([subs_considered,temp_res])
# full_res.append("##################")


# # for run in range(runs):
# #     subs_considered = random.sample(filenames, 6)+["S013"]
# #     temp_res=run_experiments(subs_considered,full_res)
# #     full_res.append([subs_considered,temp_res])
# subs_considered = ["S007","S009","S010","S018","S012","S011 ","S013"]
# temp_res=run_experiments(subs_considered,full_res)
# full_res.append([subs_considered,temp_res])
# full_res.append("##################")


# cases=[
# ['S101', 'S018', 'S004', 'S045', 'S039', 'S034', 'S018'],
# ['S093', 'S045', 'S001', 'S024', 'S069', 'S054', 'S018'],
# ['S099', 'S057', 'S091', 'S030', 'S011', 'S008', 'S018'],
# ['S025', 'S086', 'S039', 'S024', 'S075', 'S061', 'S018'],
# ['S105', 'S080', 'S064', 'S102', 'S074', 'S022', 'S018'],
# ['S007', 'S009', 'S010', 'S011', 'S012', 'S013 ', 'S018'],
# ['S079', 'S023', 'S030', 'S048', 'S033', 'S010', 'S013'],
# ['S019', 'S090', 'S039', 'S092', 'S094', 'S008', 'S013'],
# ['S099', 'S090', 'S016', 'S012', 'S086', 'S033', 'S013'],
# ['S021', 'S041', 'S018', 'S017', 'S088', 'S023', 'S013'],
# ['S020', 'S101', 'S041', 'S024', 'S071', 'S088', 'S013'],
# ['S007', 'S009', 'S010', 'S018', 'S012', 'S011 ', 'S013']


# ]

# for case in cases:
#     subs_considered = case
#     temp_res=run_experiments(subs_considered,full_res)
#     full_res.append([subs_considered,temp_res])

# pd.DataFrame(full_res).to_csv("ress.csv")


full_res=[]
#full_list=[84,106,27,23,25,45,15]
full_list=[84,106,4,98,94,16,33]
#full_list=[i for i in range(1,110)]
subs_considered = [f"S{str(e).zfill(3)}" for e in full_list if len(f"S{str(e).zfill(3)}") == 4]

temp_res=run_experiments(subs_considered,full_res)

stop=1