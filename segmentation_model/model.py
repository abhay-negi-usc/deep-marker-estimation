import torch 
import torch.nn as nn 
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # kernel=3, stride=1, padding=1 --> same convolution - same input/output width after convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # kernel=3, stride=1, padding=1 --> same convolution - same input/output width after convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),   
        )

    def forward(self, x): 
        return self.conv(x) 
    
class UNET(nn.Module): 
    def __init__(
            # self, in_channels=3, out_channels=1, features=[64,128,256,512],
            # self, in_channels=3, out_channels=1, features=[64,128,256,512,1024],
            self, in_channels=3, out_channels=1, features=[64,128,256,512,1024,2048],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Down part of UNET 
        for feature in features: 
            self.downs.append(DoubleConv(in_channels, feature)) 
            in_channels = feature 
            
        # Up part of UNET 
        for feature in reversed(features): 
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )  
            self.ups.append(DoubleConv(feature*2, feature)) 

        self.bottleneck = DoubleConv(features[-1], features[-1]*2) 

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) 

    def forward(self, x): 
        skip_connections = [] 

        for down in self.downs: 
            x = down(x) 
            skip_connections.append(x) 
            x = self.pool(x) 

        x = self.bottleneck(x) 
        skip_connections = skip_connections[::-1] 

        for idx in range(0, len(self.ups), 2): 
            x = self.ups[idx](x) 
            skip_connection = skip_connections[idx//2] 
            
            if x.shape != skip_connection.shape: 
                x = TF.resize(x, size=skip_connection.shape[2:]) # note that this padding may affect performance of segmentation at the edge of the image  
            
            concat_skip = torch.cat((skip_connection, x), dim=1) 
            x = self.ups[idx+1](concat_skip) 

        return self.final_conv(x) 

class DoubleConvWithDropout(nn.Module): 
    def __init__(self, in_channels, out_channels, dropout_prob=0.3): 
        super(DoubleConvWithDropout, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # kernel=3, stride=1, padding=1 --> same convolution - same input/output width after convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # kernel=3, stride=1, padding=1 --> same convolution - same input/output width after convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)  # Add dropout layer with the specified probability
        )

    def forward(self, x): 
        return self.conv(x)

class UNETWithDropout(nn.Module): 
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024, 2048], dropout_prob=0.3
    ):
        super(UNETWithDropout, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConvWithDropout(in_channels, feature, dropout_prob))  # Pass dropout_prob here
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConvWithDropout(feature*2, feature, dropout_prob))  # Pass dropout_prob here

        self.bottleneck = DoubleConvWithDropout(features[-1], features[-1]*2, dropout_prob)  # Pass dropout_prob here

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  # Resize to match skip_connection shape

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def test(): 
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1) 
    preds = model(x) 
    print(preds.shape) 
    print(x.shape) 
    assert preds.shape == x.shape 

if __name__ == "__main__": 
    test()
