import random
import numpy as np
import torch
from torch import optim, nn
from torch.utils import data
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from math import floor, ceil


class EncodedDataset(data.Dataset):
    encoded = []

    def __init__(self, encoded_set):
        super(EncodedDataset, self).__init__()
        self.encoded = encoded_set

    def __getitem__(self, index):
        return self.encoded[index][0], self.encoded[index][1]

    def __len__(self):
        return len(self.encoded)


class MixedDataset(data.Dataset):
    mixed = []

    def __init__(self, data_set):
        super(MixedDataset, self).__init__()
        labels_counter = torch.zeros(7)
        split_set = [[], [], [], [], [], [], []]
        for i in enumerate(data_set):
            if limited > labels_counter[i[1][1]]:
                self.mixed.append([i[1][0], i[1][1]])
                labels_counter[i[1][1]] = labels_counter[i[1][1]] + 1
                split_set[i[1][1]].append([i[1][0], i[1][1]])

        for s in split_set:
            for mix in range(int(torch.max(labels_counter).item()) - len(s)):
                img1 = s[random.randint(0, len(s)-1)][0]
                img2 = s[random.randint(0, len(s)-1)][0]
                lam = np.random.uniform(0.0, 0.2, [1])[0]
                mix_img = lam * img1 + (1 - lam) * img2
                self.mixed.append([mix_img, s[random.randint(0, len(s)-1)][1]])

    def __getitem__(self, index):
        return self.mixed[index][0], self.mixed[index][1]

    def __len__(self):
        return len(self.mixed)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=(2, 2), return_indices=True)
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=(2, 2))
        self.drp = nn.Dropout()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, encoder_layers[0], kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(encoder_layers[0]),
            nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(encoder_layers[0], encoder_layers[1], kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(encoder_layers[1]),
            nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(encoder_layers[1], encoder_layers[2], kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(encoder_layers[2]),
            nn.ReLU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(encoder_layers[2], encoder_layers[3], kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(encoder_layers[3]),
            nn.ReLU())
        self.enc5 = nn.Sequential(
            nn.Linear(encoder_layers[3] * floor(pixels/16) * floor(pixels/16), encoder_layers[4]),
            nn.BatchNorm1d(encoder_layers[4]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(encoder_layers[4], encoder_layers[5]))

        self.dec1 = nn.Sequential(
            nn.Linear(encoder_layers[5], encoder_layers[4]),
            nn.BatchNorm1d(encoder_layers[4]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(encoder_layers[4], encoder_layers[3] * floor(pixels/16) * floor(pixels/16)))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_layers[3], encoder_layers[2], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(encoder_layers[2]),
            nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_layers[2], encoder_layers[1], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(encoder_layers[1]),
            nn.ReLU())
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_layers[1], encoder_layers[0], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(encoder_layers[0]),
            nn.ReLU())
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(encoder_layers[0], 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(3),
            nn.ReLU())

    def forward(self, x):
        x = self.enc1(x)
        x, indices1 = self.mp(x)
        x = self.drp(x)

        x = self.enc2(x)
        x, indices2 = self.mp(x)
        x = self.drp(x)

        x = self.enc3(x)
        x, indices3 = self.mp(x)
        x = self.drp(x)

        x = self.enc4(x)
        x, indices4 = self.mp(x)
        x = self.drp(x)

        x = x.reshape(x.size(0), -1)
        x = self.enc5(x)
        x = nn.functional.softmax(x, dim=1)
        x = self.dec1(x)
        x = x.reshape(x.size(0), 128, int((pixels/32) * 2), int((pixels/32) * 2))

        x = self.up(x, indices4)
        x = self.dec2(x)
        x = self.drp(x)

        x = self.up(x, indices3)
        x = self.dec3(x)
        x = self.drp(x)

        x = self.up(x, indices2)
        x = self.dec4(x)
        x = self.drp(x)

        x = self.up(x, indices1)
        x = self.dec5(x)

        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.init1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Dropout()
        )
        self.init2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout()
        )
        self.init3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout()
        )
        self.bypassed4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout()
        )
        self.down_sample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(128)
        )
        self.sum5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout()
        )
        self.bypassed6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.down_sample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(256)
        )
        self.sum7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.bypassed8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.down_sample3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(512)
        )
        self.sum9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.linear10 = nn.Sequential(
            nn.Linear(int(pow((pixels/32), 2) * 512), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        x1 = self.init1(x)
        x2 = self.init2(x1)
        x3 = self.init3(x2)
        x4 = self.bypassed4(x3)
        d1 = self.down_sample1(x3)
        x5 = self.sum5((x4 * (1-down_sample_rate)) + (d1 * down_sample_rate))
        x6 = self.bypassed6(x5)
        d2 = self.down_sample2(x5)
        x7 = self.sum7((x6 * (1-down_sample_rate)) + (d2 * down_sample_rate))
        x8 = self.bypassed8(x7)
        d3 = self.down_sample3(x7)
        x9 = self.sum9((x8 * (1-down_sample_rate)) + (d3 * down_sample_rate))
        x9 = x9.reshape(x9.size(0), -1)
        x = self.linear10(x9)
        return nn.functional.log_softmax(x, dim=1)


def data_reader():
    path = './archive'
    dataset_train = ImageFolder(path, transform=Compose([Resize((pixels, pixels)), ToTensor()]))
    print(dataset_train.class_to_idx)
    dataset_train = MixedDataset(dataset_train)
    lengths = [int(floor(len(dataset_train)*0.80)), int(ceil(len(dataset_train)*0.20))]
    train_set, validation_set = torch.utils.data.random_split(dataset_train, lengths)

    train_loader_return = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader_return = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
    return train_loader_return, validation_loader_return


def train_classifier(train_set, verbose):
    model_classifier.train()
    for e in range(classifier_epoch):
        classification_loss = 0
        correct = 0
        counter = 0
        if verbose:
            print("Train set Epoch {}: ".format(e), end='')
        for batch_idx, (img, labels) in enumerate(train_set):
            optimizer_classifier.zero_grad()
            c_output = model_classifier(img)

            c_loss = nn.functional.nll_loss(c_output, labels, reduction='sum')
            classification_loss += c_loss.item()
            c_loss.backward()
            optimizer_classifier.step()

            pred = c_output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum()

            if verbose:
                prev_counter = counter
                counter = counter + len(labels)
                for i in range(int((counter/len(train_set.dataset)) * 100.) -
                               int((prev_counter/len(train_set.dataset)) * 100.)):
                    print("█", end='')
        if verbose:
            print(" Average classification loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(classification_loss / len(
                train_set.dataset), correct, len(train_set.dataset), 100. * correct / len(train_set.dataset)))


def train_encoder(train_set, verbose):
    model_encoder.train()
    for e in range(encoder_epoch):
        encoding_loss = 0
        counter = 0
        if verbose:
            print("Train set Epoch {}: ".format(e), end='')
        for batch_idx, (img, labels) in enumerate(train_set):
            optimizer_encoder.zero_grad()
            e_output = model_encoder(img)

            e_loss = nn.functional.mse_loss(e_output, img)
            encoding_loss += e_loss.item()
            e_loss.backward()
            optimizer_encoder.step()

            if verbose:
                prev_counter = counter
                counter = counter + len(labels)
                for i in range(int((counter/len(train_set.dataset)) * 100.) - int((prev_counter/len(train_set.dataset))
                                                                                  * 100.)):
                    print("█", end='')
        if verbose:
            print(" Average encoding loss: {:.4f}".format(encoding_loss/len(train_set.dataset)))


def validation_classifier(validation_set, verbose):
    model_classifier.eval()
    classification_loss = 0
    correct = 0
    score = []
    with torch.no_grad():
        for img, target in validation_set:
            c_output = model_classifier(img)
            classification_loss += nn.functional.nll_loss(c_output, target, reduction='sum').item()
            pred = c_output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            for i in range(len(img)):
                score.append([target[i], pred[i]])
    if verbose:
        print("Validation set: Average classification loss: {:.4f}, Accuracy: {}/{} " "({:.0f}%)".format(
            classification_loss / len(validation_set.dataset), correct, len(validation_set.dataset),
            100. * correct / len(validation_set.dataset)))
    output = open("results.csv", 'w')
    for i in score:
        output.write(str(i) + '\n')
    output.close()


def validation_encoder(validation_set, verbose):
    model_encoder.eval()
    encoding_loss = 0
    encoded_train = []
    with torch.no_grad():
        for img, target in validation_set:
            e_output = model_encoder(img)
            encoding_loss += nn.functional.mse_loss(e_output, img).item()
            for i in range(len(img)):
                encoded_train.append([e_output[i], target[i]])
        if verbose:
            print("Validation set: Average encoding loss: {:.4f}".format(encoding_loss/len(validation_set.dataset)))
    return torch.utils.data.DataLoader(EncodedDataset(encoded_train), batch_size=64, shuffle=True)


def run():
    train_encoder(train_loader, True)
    encoded_train = validation_encoder(train_loader, False)
    encoded_validation = validation_encoder(validation_loader, True)
    train_classifier(encoded_train, True)
    validation_classifier(encoded_validation, True)


# main
if __name__ == '__main__':
    # Loader:
    pixels = 128
    limited = 1024
    train_loader, validation_loader = data_reader()

    # Encoder:
    encoder_epoch = 4
    encoder_learning_rate = 0.001
    encoder_layers = [16, 32, 64, 128, 512, 128]
    model_encoder = Encoder()
    optimizer_encoder = optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)

    # Classifier:
    classifier_epoch = 32
    classifier_learning_rate = 0.001
    down_sample_rate = 0.3
    model_classifier = Classifier()
    optimizer_classifier = optim.Adam(model_classifier.parameters(), lr=classifier_learning_rate)

    # Runner
    summary(model_encoder, (3, pixels, pixels))
    summary(model_classifier, (3, pixels, pixels))
    print()
    run()
