import torch.nn as nn
class myNet(nn.Module):
    def __init__(self, num_classes):
        super(myNet, self).__init__()
        self.conv_layers = nn.Sequential(

            # 1st convolve-then-pool sequence
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd convolve-then-pool sequence 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd convolve-then-pool sequence 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # initilize a set of FC=>RELU layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # in_features = 128 * 28 * 28, where 28 = 224/(2*2*2)
            nn.Linear(in_features=128 * 28 * 28, out_features=256),
            nn.ReLU(),

            # initilize the softmax classifier
            nn.Linear(in_features=256, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x