from pymatreader import read_mat
import pandas as pd
import numpy as np
import ast


def load_matlab_format_data(mode='train'):
    if mode == 'train':
        LABELS_FILE = "./train/digitStruct.mat"
    if mode == 'test':
        LABELS_FILE = "./test/digitStruct.mat"

    data_mat = read_mat(LABELS_FILE)
    df = pd.DataFrame(data_mat)
    
    image_files = df.loc['name', 'digitStruct'] #data.iloc[1] 
    image_files  = np.array(image_files).reshape(-1,1)
    image_files = pd.DataFrame(image_files)
    bbox_and_labels = df.loc['bbox', 'digitStruct'] #data.iloc[0]
    bbox_and_labels = pd.json_normalize(bbox_and_labels)

    df = image_files.copy()
    df = df.join(bbox_and_labels)

    df = df.rename(columns={0: 'file_name'}) 
    df = df[['file_name', 'height', 'left', 'top', 'width', 'label']]
    
    return df
    # return image_files,bbox_and_labels

def get_labels(mode='train'):
    df = load_matlab_format_data(mode=mode)
    labels = df['label'].apply(lambda x: [int(i) for i in x] if isinstance(x, list) else int(x))
    return labels

def save_labels(df, mode='train'):
    if mode == 'train':
        LABELS_FILE = "./train/labels.csv"
    if mode == 'test':
        LABELS_FILE = "./test/labels.csv"
    
    df.to_csv(LABELS_FILE)

def convert_label_2(label):
    # If it's a list-like string, evaluate it
    if isinstance(label, str) and label.startswith('['):
        try:
            label_list = ast.literal_eval(label)  # Safely evaluate the string as a Python list
            return int(''.join(map(str, map(int, label_list))))  # Convert list to concatenated integer
        except (ValueError, SyntaxError):
            return None  # Handle any errors in conversion
    else:
        # If it's not a list, convert directly to int
        return int(float(label))

def pad_label(label, max_length=4):
    # Ensure the label is a list of integers
    if isinstance(label, int):
        label = [label]  # Convert single integer to list
    elif isinstance(label, str):
        label = ast.literal_eval(label)  # Convert string to list

    # Convert elements to integers and pad with zeros to max_length
    label = [int(l) for l in label]
    return [0] * (max_length - len(label)) + label  # Add leading zeros
    
def convert_label(label):
    # If the label is a string, parse it into a list
    if isinstance(label, str):
        label = ast.literal_eval(label)  # Safely convert string to list
    if isinstance(label, float):
        label = [int(label)]
    if isinstance(label, int):
        label = [label]
    # Convert the list elements to integers
    label = list(map(int, label))

    return pad_label(label)

def load_csv_data(mode='train'):
    if mode == 'train':
        LABELS_FILE = "./data/train/labels.csv"
    if mode == 'test':
        LABELS_FILE = "./data/test/labels.csv"

    df = pd.DataFrame(pd.read_csv(LABELS_FILE))
    df['label'] = df['label'].apply(convert_label) #.apply(lambda x: [int(i) for i in x] if isinstance(x, list) else int(x))
    
    # we can also make something like return df.iloc[0:60] to return a subset of the data
    return  df #['label']

