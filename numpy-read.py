import numpy as np

# Read the file as a binary file
with open('./sift-data/sift_groundtruth.ivecs', 'rb') as f:
    while True:
        try:
            # Read the dimension
            dimension = np.frombuffer(f.read(4), dtype=np.int32)[0]

            # Read the nearest neighbor IDs
            neighbors = np.frombuffer(f.read(4*dimension), dtype=np.int32)

            # Print the number of neighbors for this query point
            print(len(neighbors))

        except IndexError:
            # Reached end of file
            break
