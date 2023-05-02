import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# constants
TRAIN_NUM, TEST_NUM = 50000, 10000
LOSS, ERROR = 0, 1
DESIRED_ERROR = 0.2
SKIP = 5


def load_CIFAR10(batch_size, is_shuffle=True):

    # image preprocessing and data augmentation
    train_transform = transforms.Compose([
        # resize the image in order it will fit be our model.
        transforms.Resize((32, 32)),

        # randomly flips the image w.r.t horizontal axis
        transforms.RandomHorizontalFlip(p=0.5),

        # convert the image to tensor
        transforms.ToTensor(),

        # normalize the images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    test_transform = transforms.Compose([
        # resize the image in order it will fit be our model.
        transforms.Resize((32, 32)),

        # convert the image to tensor
        transforms.ToTensor(),

        # normalize the images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # download CIFAR-10 train and test datasets
    train_dataset = datasets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=train_transform,
                                  download=True)

    test_dataset = datasets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=test_transform,
                                 download=True)

    # initialize data loaders (for the train and test datasets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle)

    return train_loader, test_loader


def plot_graph(train, test, epochs, y_label):

    plt.plot(epochs, train, c='b', label="train")
    plt.plot(epochs, test, c='r', label="test")

    plt.title(f"Train {y_label} VS Test {y_label} \nas a Function of Epochs Number")
    plt.xlabel("Epochs Number")
    plt.ylabel(y_label)

    plt.legend()
    plt.show()


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # input layer image dimensions 32 X 32 x 3
        # output layer image dimensions 16 x 16 x 16
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # input layer image dimensions 16 x 16 x 16
        # output layer image dimensions 8 x 8 x 32
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # input layer image dimensions 8 x 8 x 32
        # output layer image dimensions 4 x 4 x 64
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4 * 4 * 64, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)

        # straight the image into a vector
        out = out.view(-1, 4 * 4 * 64)

        out = self.dropout(out)
        out = self.fc(out)
        predications = self.logsoftmax(out)

        return predications


def find_error_num(pred_labels, labels):
    error_num = 0
    for pred_label, label in zip(pred_labels, labels):

        if pred_label != label:
            error_num += 1

    return error_num


def find_loss_and_error(model, data_loader, loss_function, n):

    # entering evaluation mode
    model.eval()

    total_loss, total_error = 0, 0
    with torch.no_grad():
        for (images, labels) in data_loader:

            # for each image, the probability of each label to be the correct one
            pred_labels_prob = model(images)
            # the argmax of the predicated labels probabilities is the predicted label
            _, pred_labels = torch.max(pred_labels_prob.data, 1)

            # calculate the total loss and the number of errors of the predications
            total_loss += loss_function(pred_labels_prob, labels)
            total_error += find_error_num(pred_labels, labels)

    # returning to training mode
    model.train()

    loss, error = (total_loss.item() / n), (total_error / n)
    return loss, error


def train(model, train_loader, test_loader, loss_function, optimizer, epochs_num):
    train_res_lst, test_res_lst, epochs, test_error = [], [], [], 0
    for epoch in range(1, epochs_num+1):
        print(f"Epoch: {epoch}\n")

        for j, (images, labels) in enumerate(train_loader):

            # for each image, the probability of each label to be the correct one,
            # then calculate the loss
            pred_labels_prob = model(images)
            loss = loss_function(pred_labels_prob, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss, test_error = find_loss_and_error(model, test_loader, loss_function, n=TEST_NUM)

        if test_error < DESIRED_ERROR:
            train_res_lst.append(find_loss_and_error(model, train_loader, loss_function, n=TRAIN_NUM))
            test_res_lst.append((test_loss, test_error))
            epochs.append(epoch)
            break

        if epoch % SKIP == 0:
            train_res_lst.append(find_loss_and_error(model, train_loader, loss_function, n=TRAIN_NUM))
            test_res_lst.append((test_loss, test_error))
            epochs.append(epoch)

    print("Training completed successfully! \n")
    return train_res_lst, test_res_lst, epochs


def extract_res(res_lst):
    loss_lst = [res[LOSS] for res in res_lst]
    error_lst = [res[ERROR] for res in res_lst]

    return loss_lst, error_lst


def train_model_q1():
    train_loader, test_loader = load_CIFAR10(batch_size=128)
    cnn = CNN()

    print(f"\ntotal number of model parameters: {sum(param.numel() for param in cnn.parameters())} \n")

    loss_function = nn.CrossEntropyLoss(reduction="sum")
    optimiser = torch.optim.Adam(cnn.parameters(), lr=0.001)
    epochs_num = 60

    train_res_lst, test_res_lst, epochs = train(cnn, train_loader, test_loader, loss_function, optimiser, epochs_num)

    train_loss_lst, train_error_lst = extract_res(train_res_lst)
    test_loss_lst, test_error_lst = extract_res(test_res_lst)

    # error plot
    plot_graph(train_error_lst, test_error_lst, epochs, y_label="Error")

    # loss plot
    plot_graph(train_loss_lst, test_loss_lst, epochs, y_label="Cross Entropy Loss")

    # saving the trained model
    torch.save(cnn.state_dict(), "trained_model_q1.pkl")


def main():
    train_model_q1()

if __name__ == '__main__':
    main()


