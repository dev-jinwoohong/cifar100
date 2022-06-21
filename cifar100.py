from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

batch_size = 32
number_of_labels = 100
DEVICE = "cuda:0" if torch.cuda.is_available else "cpu"
transformations = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(([0.5,0.5,0.5]), ([0.5,0.5,0.5]))
                            ])

train_set = CIFAR100(root="./data", train=True, transform=transformations, download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("The number of images in a training set is: ", len(train_loader)*batch_size)

test_set = CIFAR100(root="./data", train=False, transform=transformations, download=True)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("The number of images in a test set is: ", len(test_loader)*batch_size)

superclass = ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices', 
              'household furniture', 'insects', 'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 
              'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people', 
              'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2')

classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
          'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
          'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
          'bottles', 'bowls', 'cans', 'cups', 'plates',
          'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
          'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
          'bed', 'chair', 'couch', 'table', 'wardrobe',
          'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
          'bear', 'leopard', 'lion', 'tiger', 'wolf',
          'bridge', 'castle', 'house', 'road', 'skyscraper',
          'cloud', 'forest', 'mountain', 'plain', 'sea',
          'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
          'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
          'crab', 'lobster', 'snail', 'spider', 'worm',
          'baby', 'boy', 'girl', 'man', 'woman',
          'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
          'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
          'maple', 'oak', 'palm', 'pine', 'willow',
          'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
          'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

cls = sorted(classes)


def imageshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24* 10 * 10, 100)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 2400)
        output = self.fc1(output)

        return output


model = Network().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def saveModel():
    path = "./cifar100/myFirstModel.pth"
    torch.save(model.state_dict(), path)

def testAccuracy():  
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train(num_epochs):
    
    best_accuracy = 0.0
    print("The model will be running on", DEVICE, "device")

    for epoch in range(num_epochs): 
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):            
            images = Variable(images.to(DEVICE))
            labels = Variable(labels.to(DEVICE))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


def testBatch():  
    images, labels = next(iter(test_loader))
 
    imageshow(torchvision.utils.make_grid(images))
    
    print('Real labels: ', ' '.join('%5s' % cls[labels[j]] 
                               for j in range(batch_size)))
  
    outputs = model(images)
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % cls[predicted[j]] 
                              for j in range(batch_size)))



def testClassess():
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            cls[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    train(100)
    print("Finished Training")

    testClassess()

    model = Network()
    path = "./cifar100/myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    testBatch()
