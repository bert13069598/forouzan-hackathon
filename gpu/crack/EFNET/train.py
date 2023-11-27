import torch 
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import json
from tqdm import tqdm

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), # H, W
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5)

])
parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--train", default="dataset/train", type=str, help="train folder")
parser.add_argument("--test", default="dataset/test", type=str, help="test folder")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to save")
args = parser.parse_args()
trainset = ImageFolder(root=args.train, transform=transforms, target_transform=None)
testset = ImageFolder(root=args.test, transform=transforms, target_transform=None)
labels = {i:"" for i in range(len(trainset.classes))}
print(trainset.classes)
print([trainset.classes[label] for label in labels])


from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
testloader = DataLoader(testset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)
allFiles, _ = map(list, zip(*testloader.dataset.samples))

import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
import time
import torch.optim as optim
import copy

from sklearn.metrics import f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def build_net(num_classes):
    net = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    # net.avgpool = nn.AdaptiveMaxPool2d(1)
    net.classifier = nn.Linear(1280, 2)
    net.load_state_dict(torch.load("eff_net0.pt"), strict=False)
    return net

net = build_net(len(trainset.classes))
net = net.to(device)

import os
if args.model in os.listdir():
    try:
        net.load_state_dict(torch.load(args.model))
        print(f"model loaded from {args.model}")
    except:
        print("failed to load model, creating new one")
        
criterion = nn.CrossEntropyLoss() 
optimizer = optim.AdamW(net.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.cpu() == labels.data.cpu())
            current_f1_score = f1_score(labels.data.cpu(), preds.cpu(), average='micro')

            epoch_loss = running_loss / len(trainset)
            epoch_acc = running_corrects.double() / len(trainset)

        print('Loss: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(epoch_loss, epoch_acc, current_f1_score))
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args.model)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model):
    logs = {}
    model.eval()

    total_f1_score = 0.0
    total_acc = 0.0
    counts = 0
    running_corrects = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        counts += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
       
        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size()[0]):
                # currnet file path
                logs[allFiles[batch_idx * 4 + i]] = {"pred":preds[i].item(), "true":labels[i].item()}

        # statistics
        running_corrects += torch.sum(preds.cpu() == labels.data.cpu())

        current_f1_score = f1_score(labels.data.cpu(), preds.cpu(), average='micro')
        total_f1_score += current_f1_score

        epoch_acc = running_corrects.double() / len(testset)
        total_acc += epoch_acc
    print('Acc: {:.4f} F1 Score: {:.4f}'.format(epoch_acc, current_f1_score))

    json.dump(logs, open("efnet_logs.json", "w"))
    print('Final average Acc and F1: {:4f} {:4f}'.format(total_acc/counts, total_f1_score/counts))
    return model
#%% 
model = train_model(model=net, criterion=criterion, optimizer=optimizer, num_epochs=5)
model = test_model(model)
print('Finished Training')

