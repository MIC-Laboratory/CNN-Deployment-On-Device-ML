import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import yaml
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from mobilenetv1 import MobileNetV1
from BGR2RGB565 import BGR2RGB565
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# Read the configuration from the config.yaml file
with open("config.yaml","r") as f:
    config = yaml.load(f,yaml.FullLoader)["Training_seting"]


# Read the configuration from the config.yaml file
batch_size = config["batch_size"]
training_epoch = config["training_epoch"]
num_workers = config["num_workers"]
lr_rate = config["learning_rate"]
model_width = config["model_width"]
momentum = config["momentum"]
weight_decay = config["weight_decay"]
weight_path = config["weight_path"]
best_acc = 0


# Set the dataset mean, standard deviation, and input size based on the chosen dataset
dataset_mean = [0.5,0.5]
dataset_std = [0.5,0.5]
input_size = 96

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Set the paths for dataset, weights, models, and log data
dataset_path = config["dataset_path"]


# Set the paths for dataset, weights, models, and log data
transform = transforms.Compose(
    [transforms.Resize((input_size,input_size)),
    BGR2RGB565(),
    transforms.ToTensor()])



# Load the dataset based on the chosen dataset (Cifar10, Cifar100, or Imagenet) and apply the defined transformations
print("==> Preparing data")
training_dataset = ImageFolder(root=os.path.join(dataset_path,"train"),transform = transform)
testing_dataset = ImageFolder(root=os.path.join(dataset_path,"val"),transform = transform)
# Get the number of classes in the dataset
classes = len(training_dataset.classes)


# Create data loaders for training and testing
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

# Save representive data for later on quantization
np.save("representive_data",next(iter(train_dataloader))[0].numpy())

# Create an instance of the selected model (ResNet101, MobileNetV2, or VGG) and transfer it to the chosen device
print("==> Preparing models")
print(f"==> Using {device} mode")
net = MobileNetV1(2,classes,model_width)
net.to(device)


# Define the loss function and optimizer for training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate,momentum=momentum,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epoch)

# Validation function
def validation(network,dataloader,save=True):
    # Iterate over the data loader
    global best_acc
    accuracy = 0
    running_loss = 0.0
    total = 0
    correct = 0
    network.eval()
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)


                # Perform forward pass and calculate loss and accuracy
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                pbar.update()
                pbar.set_description_str("Acc: {:.3f} {}/{} | Loss: {:.3f}".format(accuracy,correct,total,running_loss/(i+1)))
            
            # Save the model's checkpoint if accuracy improved
            if not os.path.isdir(weight_path):
                os.makedirs(weight_path)
            check_point_path = os.path.join(weight_path,"Checkpoint.pt")
            torch.save({"state_dict":network.state_dict(),"optimizer":optimizer.state_dict()},check_point_path)    
            if accuracy > best_acc:
                best_acc = accuracy
                if save:
                    PATH = os.path.join(weight_path,f"Model@MobilenetV1_ACC@{best_acc}.pt")
                    PATH = os.path.join(weight_path,f"MobilenetV1_Best.pt")
                    torch.save({"state_dict":network.state_dict()}, PATH)
                    print("Save: Acc "+str(best_acc))
                else:
                    print("Best: Acc "+str(best_acc))
    return running_loss/len(dataloader),accuracy


# Training function
def train(epoch,network,optimizer,dataloader):
    # loop over the dataset multiple times
    running_loss = 0.0
    total = 0
    correct = 0
    network.train()
    with tqdm(total=len(dataloader)) as pbar:
        # Iterate over the data loader
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()            
            
            
            accuracy = 100 * correct / total
            pbar.update()
            pbar.set_description_str("Epoch: {} | Acc: {:.3f} {}/{} | Loss: {:.3f}".format(epoch,accuracy,correct,total,running_loss/(i+1)))


# Training and Testing Loop
print("==> Start training/testing")
for epoch in range(training_epoch):
    train(epoch, network=net, optimizer=optimizer,dataloader=train_dataloader)
    loss,accuracy = validation(network=net,dataloader=test_dataloader)
    scheduler.step()
print("==> Finish")