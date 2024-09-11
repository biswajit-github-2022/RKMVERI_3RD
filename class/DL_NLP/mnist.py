# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Modification of https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Any, List, Dict

from torchsummary import summary


__all__ = [
    'AlexNet',
    'AlexNetR',
    'alexnet',
]


pretrained_model_urls = {
    'alexnetr': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
}


class AlexNet(nn.Module):
    """
    Original AlexNet:
    Input: image of size 3 x 227 x 227
    Output: number class (1000-numbers)
    """
    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 1000
    ) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.num_channels = num_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class AlexNetR(nn.Module):
    """
    Modified AlexNet:
    Input: image of size 3 x 227 x 227
    Output: number class (1000-numbers)
    """
    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 10
    ) -> None:
        super(AlexNetR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.num_channels = num_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def alexnet(
    net_type: str = 'revised',
    flag_pretrained: bool = False,
    flag_download_progress: bool = True,
    idx_model_layer: List = [0],
    idx_pretrained_model_layer: List = [0],
    **kwargs: Any
) ->AlexNetR:
    """Alexnet network architecture

    Args:
        net_type (str): alexnet varients ('original', 'revised')
        flag_pretrained (bool): False - if "True" initialize with pre-trained model weights
        flag_download_progress: True - progress bar will show during pre-trained model download
        idx_model_layer (List): [0] - adjust 0-th layer's wights
        idx_pretrained_model_layer (List): [0] - use 0-th layer's wights of pre-trained model to adjust the current model
    """
    if net_type=='original':
        model = AlexNet(**kwargs)
    else:
        model = AlexNetR(**kwargs)
        if flag_pretrained:
            dict_pretrained_state = load_state_dict_from_url(pretrained_model_urls['alexnetr'], progress=flag_download_progress)
            print('-'*70)
            print('Using weight from the pretrained model from: {}' .format(pretrained_model_urls['alexnetr']))
            print('-'*70)
            dict_model = weight_transform_layer_pos(model.state_dict(), dict_pretrained_state, idx_model_layer, idx_pretrained_model_layer)
            print('-'*70)
            model.load_state_dict(dict_model)

    return model


def weight_transform_layer_pos(
    dict_model: Dict[str, torch.Tensor],
    dict_pretrained_state: Dict[str, torch.Tensor],
    idx_model_layer: List[int] = [0],
    idx_pretrained_model_layer: List[int] = [0]
) ->Dict:
    """Weights update of the contom layers based on the pre-trained weights on ImageNet

    Args:
        dict_model (state-dict): new (custom) model weights state dict
        dict_pretrained_state (state-dict): pretrained model weights state dict
        idx_model_layer (list of intergers): index of the new (custom) model state dict want to change
        idx_pretrained_model_layer (list of intergers): index of the pre-trained model state dict from where weights will be replaced
    """
    # first copy all weights with same name and size
    for k, v in dict_pretrained_state.items():
        if k in dict_model:# matched weights name
            if len(v.shape)==len(dict_model[k].shape):# matched weights size
                dims_same = 1
                for i in range(len(v.shape)):
                    if v.shape[i]!=dict_model[k].shape[i]:
                        dims_same = 0
                if dims_same:
                    print('weight: model[{}] <= pretrained_model[{}]' .format(k, k))
                    dict_model[k] = v
    # adjust the weights of the custom layers (by mean of the pretrained weights)
    for idx in range(len(idx_model_layer)):
        key_model = list(dict_model.keys())[idx_model_layer[idx]]
        key_pretrained_model = list(dict_pretrained_state.keys())[idx_pretrained_model_layer[idx]]
        print('weight: model[{}] <= pretrained_model[{}]' .format(key_model, key_pretrained_model))
        w_org = dict_model[key_model]
        w_trans = dict_pretrained_state[key_pretrained_model]

        if w_org.shape[1] != w_trans.shape[1]:

            w_trans = w_trans.mean(axis=1)
            for i in range(w_org.shape[1]):
                w_org[:,i,:,] = w_trans
        dict_model[key_model] = w_org

    return dict_model


# if __name__ == '__main__':

#     num_channels = 1
#     H, W = 227, 227
#     num_classes = 10
#     batch_size = 5
#     model = alexnet(flag_pretrained=True, num_channels=num_channels, num_classes=num_classes)

#     print('-'*70)
#     print('Network architechture (num channels-{}, num classes- {}) as follows:' .format(num_channels, num_classes))
#     print('-'*70)
#     print(model)
#     print('-'*70)

#     print('Network summary:')
#     print('-'*70)
#     summary(model, (num_channels, H, W, ), device=str("cpu"))
#     print('-'*70)

#     print('Network input output dims check')
#     print('-'*70)
#     x = torch.randn(batch_size, num_channels, H, W)
#     y = model(x)
#     print('input shape(batch_size x num_channels x height x width): {}\noutput shape(batch_size x num_classes): {}' .format(x.shape, y.shape))
#     print('-'*70)





# transformation and loading dataset from torchvision datasets
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((227,227), antialias=True)])
train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
# Create an instance of the image classifier model
device = torch.device("cpu")
classifier =  alexnet(flag_pretrained=True, num_channels=1, num_classes=10).to(device)

# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


# Train the model
for epoch in tqdm(range(10)):  # Train for 10 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = classifier(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")

# Save the trained model
torch.save(classifier.state_dict(), 'model_state.pt')

# Load the saved model
with open('model_state.pt', 'rb') as f: 
     classifier.load_state_dict(load(f))  

# Perform inference on an image
img = Image.open('image.jpg')
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img).unsqueeze(0).to(device)
output = classifier(img_tensor)
predicted_label = torch.argmax(output)
print(f"Predicted label: {predicted_label}")



