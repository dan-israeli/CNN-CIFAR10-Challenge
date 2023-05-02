# imports
from train_model import CNN, load_CIFAR10, find_loss_and_error, TEST_NUM
import torch
import torch.nn as nn


def evaluate_model_q1():
    _, test_loader = load_CIFAR10(batch_size=128)

    cnn = CNN()
    cnn.load_state_dict(torch.load("trained_model_q1.pkl"))
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    # entering evaluation mode
    cnn.eval()

    _, test_error = find_loss_and_error(cnn, test_loader, loss_function, n=TEST_NUM)

    # the error of the trained model
    print(f"trained model test error: {round(test_error, 3)} \n")


def main():
    evaluate_model_q1()


if __name__ == '__main__':
    main()
