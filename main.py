import argparse
import torch
from Models.alexnet import AlexNet
from Models.resnet import *
from Models.models import *
from torch import nn
from utils import read_data_and_dataloader, save_model


def main(dataset, network, train_batch_size, evaluate_batch_size, learning_rate, epochs, optimizer, times, gpu):
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:" + str(gpu))
    train_dataloader, test_dataloader, train_data, test_data = read_data_and_dataloader(dataset, train_batch_size, evaluate_batch_size)
    test_data_size = len(test_data)
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    for i in range(times):
        print("---------------Running time:{}---------------".format(i))
        model = train_model(model, device, epochs, train_dataloader, optimizer, loss_fn)
        # 测试步骤开始
        test_model(model, device, test_dataloader, loss_fn, test_data_size)
        save_model(model, dataset, network, i)
        print("Model Saved!")


def train_model(model, device, epochs, train_dataloader, optimizer, loss_fn):
    # training process
    model.train()
    for epoch in range(1, epochs + 1):
        print("********epoch:{}********".format(epoch))
        total_train_step = 0
        model.train()
        for (inputs, targets) in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("training step {}, Loss: {}".format(total_train_step, loss.item()))
    return model


def test_model(model, device, test_dataloader, loss_fn, test_data_size):
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for (inputs, targets) in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("Test Loss: {}".format(total_test_loss))
    print("Test Accuracy: {}".format(total_accuracy / test_data_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10",choices=["MNIST", "synthetic", "Cifar10", "Cifar100", "SVHN", "FashionMNIST"])
    parser.add_argument("--network", type=str, default="ResNet")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--evaluate_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Dataset : {}".format(args.dataset))
    print("Network : {}".format(args.network))
    print("Train Batch size : {}".format(args.train_batch_size))
    print("Evaluate Batch Size : {}".format(args.evaluate_batch_size))
    print("Learing rate : {}".format(args.learning_rate))
    print("Epochs : {}".format(args.epochs))
    print("Optimizer : {}".format(args.optimizer))
    print("Times : {}".format(args.times))
    print("GPU : {}".format(args.gpu))
    print("=" * 80)

    main(args.dataset, args.network, args.train_batch_size, args.evaluate_batch_size, args.learning_rate, args.epochs, args.optimizer, args.times, args.gpu)
