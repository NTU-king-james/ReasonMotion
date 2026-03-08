
import pickle
import sys

path = "/home/allen/datasets/FineFS_5s/3_final/valid/4F/4F_0011/new_res.pk"
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print("Type:", type(data))
        if isinstance(data, dict):
            print("Keys:", data.keys())
            # Print shapes if they are arrays
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"Key: {k}, Shape: {v.shape}")
                else:
                    print(f"Key: {k}, Type: {type(v)}")
        else:
            print("Data:", data)
except Exception as e:
    print("Error:", e)
