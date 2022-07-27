import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import os
import datetime

def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(value)

def average_last_data(data_list, data_count: int = -1) -> list:
    """ Returns a list containing averaged values of last n data from the data list.

    Args:
        data_list (ArrayLike): data list
        data_count (int, optional): data count n. if it's a negative value, data_count is len(data_list). Defaults to -1.

    Returns:
        list: a list containing averaged values
    """
    
    data_list = np.array(data_list)
    averages = []
    if data_count < 0:
        data_count = len(data_list)
        
    for i in range(len(data_list)):
        s = max(i - data_count + 1, 0)
        e = i + 1
        averages.append(np.mean(data_list[s:e]))
        
    return averages

def add_datetime_suffix(basename: str, delimiter: str = '_') -> str:
    """ Add a datetime suffix wtih delimiter to the basename. (e.g. basename_220622_140322) """
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    return delimiter.join([basename, suffix])

def create_dir(directory):
    """ If there's no directory, create it. """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        