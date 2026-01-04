import torch 
import torch.nn as nn 
import torchvision.transforms.functional as TF 
"""
We will use padding = 1 (to match input and output shapes) and add batch normalization (not in the original paper)
"""
class DoubleConv(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int) -> None: 
        super().__init__() #initializes the hidden attributes such as params, modules etc from the base class
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),#When using BatchNorm, the bias term in Convolution is redundant (BatchNorm calculates its own mean/bias subtraction)
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )

    def forward(self, x): 
        return self.conv(x)
    

class Unet(nn.Module): 
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]): 
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList() 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of the Unet(encoder) 
        for feature in features: 
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up Part of Unet(Decoder)
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
        
        #Encoder pass 
        for down in self.downs: 
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] #reversed the list for easy iterations  

        #Decoder pass 
        for idx in range(0, len(self.ups), 2): #step value is 2 becasuse we have an upconv and then double conv
            x = self.ups[idx](x) #upconv
            skip_connection = skip_connections[idx//2] #get corresponding skip connection

            #resize if shapes don't match (e.g. if input size was not divisible by 16)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate along channel dimension (dim=1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # DoubleConv
        
        return self.final_conv(x)