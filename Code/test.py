import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    dist = np.loadtxt("10-1-dist.txt")
    jobs = np.loadtxt("10-1-job.txt")
    
    scalar = MinMaxScaler()
    jobs = scalar.fit_transform(jobs)
    scaled_dist = np.array(dist).flatten().reshape(-1, 1)
    scaled_flat = scalar.fit_transform(scaled_dist)
    dist = scaled_flat.reshape(dist.shape)
    print("1", dist)

if __name__ == "__main__":
    main()