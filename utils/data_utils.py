import csv

def save_landmarks(filename, landmark_list, label):
    row = [label] + [coord for lm in landmark_list for coord in (lm.x, lm.y, lm.z)]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def load_dataset(filename):
    import pandas as pd
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y
