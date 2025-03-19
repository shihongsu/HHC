from torch.distributions import Categorical
import torch

def main():

    logits = torch.tensor([[1, 2, 3, 4],
                            [1, 2, 3, float('-inf')]])

    dist = Categorical(logits = logits)

    print(dist.sample())

if __name__ == "__main__":
    main()