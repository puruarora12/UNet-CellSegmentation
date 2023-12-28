from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os
import wandb
import time
#import any other libraries you need below this line

# Paramteres

# learning rate
lr = 0.001 
epoch_n = 50  
image_size = 360 
root_dir = os.getcwd()
# training batch size
batch_size = 1  # 4
# use checkpoint model for training
load = False
# use GPU for training
gpu = True

augment_data = True
wandb.init( 
    project="UNet Cell Segmentation", 
    config={
        "learning_rate": lr,
        "epochs": epoch_n,
        "batch_size": batch_size,
        "image_size": image_size
    }
)

data_dir = os.getcwd()+ '/data/cells'

trainset = Cell_data(data_dir=data_dir, size=image_size, augment_data=augment_data)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False, augment_data=augment_data)
testloader = DataLoader(testset, batch_size=batch_size)

device = torch.device('cuda:0' if gpu else 'cpu')

model = UNet().to('cuda:0').to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load('checkpoint.pt'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=0.0005)  

train_loss_log = []
test_loss_log = []

model.train()
begin = time.time() 
for e in range(epoch_n):
    epoch_loss = 0
    correct_train = 0
    total_train = 0
    model.train()
    for i, data in enumerate(trainloader):
        image, label = data

        # image = image.unsqueeze(1).to(device)
        image = image.to(device)
        label = label.long().to(device)

        pred = model(image)
        print(pred.shape)
        label = label.squeeze(1)

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

        loss = criterion(pred, label)
        # print(loss)
        total_train += label.shape[0] * label.shape[1] * label.shape[2]
        
        _, pred_labels_train = torch.max(pred, dim = 1)
        correct_train += (pred_labels_train == label).sum().item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    t_acc = correct_train / total_train
    train_loss = epoch_loss / trainset.__len__()
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    train_loss_log.append(epoch_loss / trainset.__len__())

    torch.save(model.state_dict(), 'checkpoint.pt')
    # torch.onnx.export(model, sample_image, 'UNet_segments', input_names='Cells', output_names='Masks')
    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            # image = image.unsqueeze(1).to(device)
            image = image.to(device)
            label = label.long().to(device)

            pred = model(image)

            label = label.squeeze(1)

            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()
        v_acc = correct/total
        v_loss = total_loss/testset.__len__()
        print('Accuracy: %.4f ---- Loss: %.4f' % (v_acc, v_loss))

        test_loss_log.append(total_loss / testset.__len__())
        if correct/total > 0.75:
              torch.save(model.state_dict(), 'checkpoint.pt')
        
    wandb.log({'train_accuracy': t_acc,'time': time.time()-begin, 'training_loss': train_loss, 'validation_loss': v_loss})
    wandb.watch(model)

model.eval()

testset = Cell_data(data_dir=data_dir, size=572, train=False, augment_data=False)
testloader = DataLoader(testset, batch_size=batch_size)

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)

        # input_image = image.unsqueeze(0).unsqueeze(0).to(device)
        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        labels = labels.squeeze(0)
        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
       
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y]

        labels = labels.numpy()
        output_masks.append(output_mask)
        output_labels.append(labels)

# Plot usingplt plot train-test plot

plt.plot(range(epoch_n), train_loss_log, 'g', label='Training Loss')
plt.plot(range(epoch_n), test_loss_log, 'r', label='Testing Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train-Test Loss')
plt.legend()
plt.show()
plt.savefig("traintestloss.png")

fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))

for i in range(testset.__len__()):
  axes[i, 0].imshow(output_labels[i])
  axes[i, 0].axis('off')
  axes[i, 1].imshow(output_masks[i])
  axes[i, 1].axis('off')
fig.savefig("visual.png")