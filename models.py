import torch
import torch.nn as nn
from torch.nn import functional as F
from  model.resnet import ResNet18
from torchvision.models import efficientnet_b0
from torchvision.models import mobilenet_v3_small



class MLP(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(arg.resize* arg.resize * 1, arg.resize),
            nn.ReLU(),
            nn.Linear(arg.resize, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
       
    def forward(self, x):
        
        return self.layers(x)
    
    
class MLP_meta(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(arg.resize* arg.resize * 1, arg.resize),
            nn.ReLU(),
            nn.Linear(arg.resize, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )
        self.linear1 = nn.Linear(11, 11)
        self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(20, 1)
        # self.relu2 = nn.ReLU()
        self.output = nn.Linear(22, 1)
    def forward(self, x, meta):
        x_img = self.layers(x)
    
        x_meta = self.linear1(meta)
        x_meta = self.relu1(x_meta)
        x = torch.concat([x_img, x_meta],dim=1)
        x = self.output(x)
        
        return x
    
class CNN_Base(torch.nn.Module):
    
    def __init__(self, args):
        super(CNN_Base, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride = 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 128x128x64 inputs -> 64*64 outputs ->1로 줄임
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32*100*100 , 128 , bias=True)
        self.fc2 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.crop(out)
        out = self.flatten(out)
#print( out.shape ) 

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
  
class CNN(torch.nn.Module):
    
    def __init__(self, args):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride = 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 128x128x64 inputs -> 64*64 outputs ->1로 줄임
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32*(args.resize//4)//2*(args.resize//4)//2, 16*16, bias=True)
        self.fc2 = torch.nn.Linear(16*16, 1, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.crop(out)
        out = self.flatten(out) 
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    
    def crop(self, x):
        w_mid, h_mid = x.size(1)//2, x.size(2)//2
        width = 16
        height = 16
        return x[:, w_mid-width:w_mid+width, h_mid-height:h_mid+height]
        # return x[:,x.size(1)//4:(x.size(1)//4)*3,x.size(2)//4:(x.size(2)//4)*3]
    
    
def get_net():
    #resnet = models.resnet34(pretrained=True)
    #resnet.fc = torch.nn.Linear(512, 1)
    resnet = ResNet18(1)    
    
    return resnet


class CNN_CenterCrop_10(torch.nn.Module):
    
    def __init__(self, args):
        super(CNN_CenterCrop_10, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride = 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 128x128x64 inputs -> 64*64 outputs ->1로 줄임
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32*10*10, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.crop(out)
        out = self.flatten(out)
#print( out.shape ) 

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

    def crop(self, x):
        w_mid, h_mid = x.size(2)//2, x.size(3)//2

        width = 5
        height = 5
        return x[: , : , w_mid-width:w_mid+width, h_mid-height:h_mid+height]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args.resize* args.resize * 1, args.resize),
            nn.ReLU(),
            nn.Linear(args.resize, 128),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32*(args.resize//4)*(args.resize//4)+128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        conv = conv.view(conv.size(0), -1)
        mlp = self.layer1(x)
        conv_mlp = torch.cat([conv,mlp], dim=1)
        conv_mlp = self.fc_layers(conv_mlp)

        return conv_mlp



class ResNet_CenterCrop(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5, crop_size=2):
        super(ResNet_CenterCrop, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.reg1 = nn.Linear( 4*4*512, 512)
        self.reg2 = nn.Linear( 512, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()
        self.classifer = nn.Linear(512, 1)

        self.crop_size = crop_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def crop(self, x):
        w_mid, h_mid = x.size(2)//2, x.size(3)//2

        width = self.crop_size
        height = self.crop_size
        return x[:, : ,  w_mid-width:w_mid+width, h_mid-height:h_mid+height]
 

    def forward(self, x):
        
        #import pdb 
        #pdb.set_trace() 
        #print( x.shape )
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #fe_map = out
        out = self.crop(out)


        out = self.flatten(out)
        out = self.reg1(out)
        out = self.reg2(out)

 
        return out  #, fe_map


def ResNet18_CenterCrop(args):
    return ResNet_CenterCrop(BasicBlock, [2, 2, 2, 2], 1,args.crop_size)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.reg = nn.Linear(512, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        #self.classifer = nn.Linear(512, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        #import pdb 
        #pdb.set_trace() 
        #print( x.shape )
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #fe_map = out
        #print( out.shape)
        out = self.avg_pool(out)
        #print( out.shape)
        out = out.view(out.size(0), -1)
        #print( out.shape)
        reg = self.reg(out)
        #out = self.classifier(out)

        #print( out.shape)
 
        return reg  #, fe_map




def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)



def efficient_b0(args):
    net = efficientnet_b0()
    net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1, bias=True)
        )
    return net

def mobile_v3(args):
    net = mobilenet_v3_small()
    net.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    net.classifier = nn.Linear(in_features=576, out_features=1, bias=True)
    
    return net


class Efficient_CenterCrop(nn.Module):
    def __init__(self, args,crop_size=2):
        super(Efficient_CenterCrop, self).__init__()
        self.encoder = efficientnet_b0().features
        self.encoder[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.crop_size = crop_size
        self.flatten = nn.Flatten()
        self.regression =  nn.Linear(1280*self.crop_size*2*self.crop_size*2, out_features=1, bias=True)
        
    def crop(self, x):
        w_mid, h_mid = x.size(2)//2, x.size(3)//2

        width = self.crop_size
        height = self.crop_size
        return x[:, : ,  w_mid-width:w_mid+width, h_mid-height:h_mid+height]

    def forward(self, x):
        x = self.encoder(x)
        x = self.crop(x)
        x = self.flatten(x)
        x = self.regression(x)
        
        return x
    
    
class Mobilenet_CenterCrop(nn.Module):
    def __init__(self, args,crop_size=2):
        super(Mobilenet_CenterCrop, self).__init__()
        self.encoder = mobilenet_v3_small().features
        self.encoder[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.crop_size = crop_size
        self.flatten = nn.Flatten()
        self.regression =  nn.Linear(576*self.crop_size*2*self.crop_size*2, out_features=1, bias=True)
        
    def crop(self, x):
        w_mid, h_mid = x.size(2)//2, x.size(3)//2

        width = self.crop_size
        height = self.crop_size
        return x[:, : ,  w_mid-width:w_mid+width, h_mid-height:h_mid+height]

    def forward(self, x):
        x = self.encoder(x)
        x = self.crop(x)
        x = self.flatten(x)
        x = self.regression(x)
        
        return x