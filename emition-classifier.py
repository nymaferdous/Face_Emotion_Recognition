import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from resnet import ResNet18
from models import ResNet18 as net
from vgg import vgg16 as vggnet
import densenet
from alexnet import AlexNet as alex
import conv
from torchsummary import summary

# data_dir = '/home/nyma/Desktop/FDatawork'
data_dir='/home/nyma/Desktop/CK+48'
feature_extract = True

def load_split_train_test(datadir, valid_size=0):
    train_transforms = transforms.Compose([transforms.Resize(192),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(192),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    num_test = len(test_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=12)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=12)
    # print(len(trainloader))
    # print(len(testloader))

    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, 0)

# a = 0
# for idx, (inputs, labels) in enumerate(trainloader):
#     lbl = (labels==0).sum()
#     a += lbl.item()
#     print(idx, a)
#
# print("-----------------------")
#
#
# a = 0
# for idx, (inputs, labels) in enumerate(testloader):
#     lbl = (labels==0).sum()
#     a += lbl.item()
#     print(idx, a)
# exit()


# print("Classes", len(trainloader.dataset))


def train():
    # exit()
    model.train()
    display_freq = 10
    avg_loss = 0
    avg_acc = 0
    for iter, (inputs, labels) in enumerate(trainloader):

        # input = inputs.to(device)
        # out = model(input)
        # print("Outside: input size", input.size(),
        #       "output_size", out.size())
        # print("Labels",labels)
        ####################################################
        # img = inputs[0]
        # img = img.detach().numpy().transpose(1,2 , 0)
        # img = (img-img.min())/(img.max()-img.min())
        # import matplotlib.pyplot as plt
        # plt.imshow(img)  output = model(input)
        #         print("Outside: input size", input.size(),
        #               "output_size", output.size())
        # plt.show()

        # 1: zero out the gradients
        optimizer.zero_grad()

        # summary(model_single, (3, 224, 224))


        # 2: compute the loss
        inputs, labels = inputs.cuda(), labels.cuda()
        preds = model.forward(inputs)
        loss = criterion(preds, labels)

        pred_lbl = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred_lbl.eq(labels.view_as(pred_lbl)).type(torch.FloatTensor).mean().item()


        # 3: backpropagate the loss
        loss.backward()

        # 4: update weights
        optimizer.step()

        avg_loss += loss.item()
        avg_acc += acc

        if iter % display_freq == 0 and iter>0:
            avg_loss /= display_freq
            avg_acc /= display_freq
            lear= scheduler.get_lr()
            print('epoch: %02d, iter: %2d, xent loss: %.4f, train acc: %.4f LR:%7s' % (epoch, iter, avg_loss, avg_acc, lear))

            avg_acc = 0
            avg_loss = 0


# def calc_accuracy(model):
#     model.eval()
#     model.to(device='cuda')
#     n_corrects = 0
#     n_total_samples = 0
#     with torch.no_grad():
#         print(testloader)
#         for iter, (inputs, labels) in enumerate(testloader):
#
#             inputs, labels = inputs.cuda(), labels.cuda()  # obtain the outputs from the model
#
#             outputs = model(inputs)  # max provides the (maximum probability, max value)
#             pred_lbl = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             n_corrects += pred_lbl.eq(labels.view_as(pred_lbl)).type(torch.FloatTensor).sum().item()
#             n_total_samples += inputs.size(0)
#             if iter % 10 == 0:
#                 print(iter, '/', len(testloader), n_corrects/n_total_samples)
#     print("test accuracy: %.4f" % (n_corrects/n_total_samples))


train_losses, test_losses = [], []

    # train()
# model_single = AlexNet()
# model_single = net(n_classes=7)
model = ResNet18()
for param in model.parameters():
    param.requires_grad = True

# model_single= vggnet()
# model_single = models.densenet161(pretrained=False)
# model_single= densenet.
# model = nn.DataParallel(model_single)
# model.cuda()
# model = load_model('finger_model_epoch-90.pth')
trainloader, testloader = load_split_train_test(data_dir, .2)
criterion = nn.CrossEntropyLoss()


# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=5e-4)
model.cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.05, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# print("Number",len(testloader.dataset))
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model = models.resnet18(pretrained=False)
# for param in model.parameters():
#     param.requires_grad = True
#
# model.fc = nn.Sequential(nn.Linear(512, 29500))
# nn.ReLU(),
# nn.Dropout(0.2),
# nn.Linear(512, 29500))

epochs = 400
steps = 0
running_loss = 0
print_every = 10
training = True

model.load_state_dict(torch.load('/home/nyma/PycharmProjects/classifiertorch/emotion_model__low_epoch-80-resnet18.pth',map_location=lambda storage, loc: storage))
if training:
    for epoch in range(epochs):
        scheduler.step()
        train()
        if epoch!= 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), 'emotion_model__epoch-{}-resnet18.pth'.format(epoch))
            print("model is saved successfully!")
    torch.save(model.state_dict(), 'emotion_model-resnet.pth')

        # exit()
else:
    model.load_state_dict(torch.load('emotion_model_epoch-90.pth'))
    calc_accuracy(model)

# plt.plot(train_losses, label='Training loss')
# # plt.plot(test_losses, label='Validation loss')
# plt.legend(frameon=False)
# # plt.show()
