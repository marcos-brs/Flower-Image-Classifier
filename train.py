import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

def get_parser():
    parser = argparse.ArgumentParser(description='Flower Classifcation trainer')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: vgg11, vgg16, vgg19]')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='hidden units')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    
    parser.add_argument('--save_dir' , type=str, default='my_checkpoint.pth', help='path of your saved model')
    return parser.parse_args()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def conf_train_model(data_dir, device, dataloaders, dataset_sizes, class_size):
    
    args = get_parser()
    
    if args.arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        return

    # Freeze the params
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[0].in_features

    classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),    
        nn.Linear(args.hidden_units, int(args.hidden_units/2)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_units/2), class_size),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learn_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, args.epochs)
    
    return model, optimizer, exp_lr_scheduler

def save_model(model, optimizer, exp_lr_scheduler, image_datasets, path):
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'scheduler': exp_lr_scheduler.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
    }
    torch.save(checkpoint, path)

def main():
    
    args = get_parser()
           
    data_dir = args.data_dir
    gpu = args.gpu
    save_dir = args.save_dir
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    input_shape = 224
    batch_size = 32
    scale = 256

    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(scale),
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    class_size = len(class_names)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    model, optmizer, exp_lr_scheduler = conf_train_model(data_dir, device, dataloaders, dataset_sizes, class_size)
        
    save_model(model, optmizer, exp_lr_scheduler, image_datasets, save_dir)

if __name__ == "__main__":
    main()