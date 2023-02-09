import torch
from numpy import mean
import sys

def main():
    args = sys.argv[1:]
    state = torch.load(args[0])
    print(f'the final accuracy is {mean(state["accs"])}.')

if __name__ == "__main__":
    main()
