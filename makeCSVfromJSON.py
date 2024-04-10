import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import json

data = [
        'Price',
        'RAM_Price',
        'RAM_DDR_Type',
        'RAM_Frequency', 
        'RAM_Volume', 
        "CPU_Price",
        "CPU_Performance_coef", 
        "CPU_Core_count",
        "CPU_SMT",
        'CPU_Boost_clock',
        'CPU_TDP',
        "GPU_Price", 
        "GPU_Performance_coef",
        "GPU_VRAM",
        'GPU_Boost_clock',
        # 'Storage_Capacity'
        ]
with open('EXTENDED_PC.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(data)

with open('cpu.json', 'r') as file_1:
    cpuJSON = json.load(file_1)
with open('memory.json', 'r') as file:
    ramJSON = json.load(file)
with open('gpu.json', 'r') as file:
    gpuJSON = json.load(file)
with open('storage.json', 'r') as file:
    storageJSON = json.load(file)

cpu_df = pd.read_json('cpu.json')
cpu_max_price = cpu_df['price'].dropna().max()

gpu_df = pd.read_json('gpu.json')
gpu_max_price = gpu_df['price'].dropna().max()

# NZXT H5 Flow, price = 95
casePrice = 95 
# Любая материнка за price = 150
motherboardPrice = 150

# Corsair RM750e (2023), price = 100
powerSupplyPrice = 100

# Любой кулер процессора за price = 40
cpuCoolerPrice = 40

storagePrice = 50

with open('EXTENDED_PC.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for i in range(0, 1000):

        cpuPrice = cpuJSON[i]['price']
        gpuPrice = gpuJSON[i]['price']
        ramPrice = ramJSON[i]['price']
        # storagePrice = storageJSON[i]['price']

        if (cpuPrice == None): cpuPrice = 200
        if (gpuPrice == None): gpuPrice = 666
        if (ramPrice == None): ramPrice = 70
        if (ramPrice > 500):continue

        # if (storagePrice == None): storagePrice = 50

        price = cpuPrice + gpuPrice + ramPrice + storagePrice + casePrice + motherboardPrice + powerSupplyPrice + cpuCoolerPrice
        price = round(price, 2)

        # if price > 3000: continue
        if price == 2164.97: continue
        if price == 2340.48: continue
        if price == 2433.98: continue
        if price == 2431.48: continue
        # if price > 1290 and price < 1350 : continue

        # if CPU_Price + GPU_Price + RAM_Price + Storage_Capacity > 3000: continue

        RAM_Price = ramJSON[i]['price']
        # if (RAM_Price > 300): RAM_Price = 250

        RAM_DDR_Type = ramJSON[i]['speed'][0]
        RAM_Frequency = ramJSON[i]['speed'][1]
        RAM_Volume = ramJSON[i]['modules'][0]*ramJSON[i]['modules'][1]

        CPU_Price = cpuJSON[i]['price']
        if (CPU_Price == None): CPU_Price = 200

        CPU_Performance_coef = ((CPU_Price/cpu_max_price)*10).round(2)
        CPU_Core_count = cpuJSON[i]['core_count']
        # Если CPU_SMT = 0 - гипертрейдинга нет, если CPU_SMT = 1 - гипертрейдинг есть
        CPU_SMT = 0
        if cpuJSON[i]['smt'] == True: 
            CPU_SMT = 1
        CPU_Boost_clock = cpuJSON[i]['boost_clock']
        CPU_TDP = cpuJSON[i]['tdp']

        # gpuPrice = gpuJSON[i]['price']
        GPU_Performance_coef = ((gpuPrice/gpu_max_price)*10).round(2)
        

        GPU_VRAM = gpuJSON[i]['memory'] 

        GPU_Boost_clock = gpuJSON[i]['boost_clock']

        # Storage_Capacity = storageJSON[i]['capacity']

        row =   [
                price,
                RAM_Price,
                RAM_DDR_Type,
                RAM_Frequency,
                RAM_Volume,
                CPU_Price,
                CPU_Performance_coef,
                CPU_Core_count,
                CPU_SMT,
                CPU_Boost_clock,
                CPU_TDP,
                gpuPrice,
                GPU_Performance_coef,
                GPU_VRAM,
                GPU_Boost_clock,
                # Storage_Capacity,
                ]
        writer.writerow(row)

# Read the CSV file into a DataFrame
df = pd.read_csv('EXTENDED_PC.csv')

# Replace missing values with the median
for column in df.columns:
    median = df[column].median()
    df[column].fillna(median, inplace=True)

# Write the updated data back to the CSV file
df.to_csv('EXTENDED_PC.csv', index=False)