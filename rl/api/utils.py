import datetime

def str_to_time(str):
    return datetime.datetime.strptime(str, "%Y/%m/%d-%H:%M:%S")

def name_to_time(str1):
    str1 = str1.split("_")[2]   
    return str_to_time(str1)
