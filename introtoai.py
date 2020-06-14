# coding=gb2312
#参考：https://blog.csdn.net/jizhidexiaoming/article/details/96485095
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

DEVICE = 'cuda' if torch.cuda.is_available()==True else 'cpu' #cuda是否可用
#os.environ['CUDA_VISIBLE_DEVICE']=1  #设定gpu
if not os.path.exists('./img'):
    os.mkdir('./img')
if not os.path.exists('./testres'):
    os.mkdir('./testres')
    
batch_size = 128
num_epoch = 1000
fake_dimension = 100 #随机输入维数
criterion = nn.BCELoss()  # 二分类交叉熵loss

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) # 图像预处理，变换到[-1,1]
 
# mnist数据集下载
mnist = datasets.MNIST(root='./data/', train=True, transform=img_transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True) #加载数据

mnist_test = datasets.MNIST(root='./data/', train=False, transform=img_transform, download=True)
testloader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)

class generator(nn.Module): #生成器
    def __init__(self):
        super(generator, self).__init__()
        self.n1 = nn.Sequential(nn.Linear(100,256),nn.ReLU(True)) #全连接网络，ReLU激活函数
        self.n2 = nn.Sequential(nn.Linear(256,256),nn.ReLU(True))
        self.n3 = nn.Sequential(nn.Linear(256,784),nn.Tanh())
        #self.n4 = nn.Sequential(nn.Linear(512,784),nn.Tanh())
 
    def forward(self, x):  #前向传播
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)
        #x = self.n4(x)
        return x

class discriminator(nn.Module):  #判别器
    def __init__(self):
        super(discriminator, self).__init__()
        self.n1 = nn.Sequential(nn.Linear(784,256),nn.LeakyReLU(0.2))
        self.n2 = nn.Sequential(nn.Linear(256,256),nn.LeakyReLU(0.2))
        self.n3 = nn.Sequential(nn.Linear(256,1),nn.Sigmoid())
        #self.n4 = nn.Sequential(nn.Linear(128,1),nn.Sigmoid())
 
    def forward(self, x):  #前向传播
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)
        #x = self.n4(x)
        return x

def convert2img(x):  #转换成图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  
    out = out.view(-1, 1, 28, 28)  # 变成需要的尺寸
    return out

#D = discriminator()
#G = generator() 
#D = D.to(DEVICE)
#G = G.to(DEVICE) 

def train(D,G):
    Doptimizer = torch.optim.Adam(D.parameters(), lr=0.0003)  # 使用 Adam优化器
    Goptimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    for epoch in range(num_epoch):  # 训练
        for i, (img, _) in enumerate(dataloader):
            num = img.size(0)

            img = img.view(num, -1)  # view可以改变张量尺寸，此处将图片展开为28*28=784
            real_img = Variable(img).cuda()  
            real_label = Variable(torch.ones(num)).cuda()  # 定义真实图片label为1
            fake_label = Variable(torch.zeros(num)).cuda()  # 定义假图片的label为0
            fake = Variable(torch.randn(num, fake_dimension)).cuda()  # 生成随机输入：均值0，方差1
            
            #判别器训练
            Doptimizer.zero_grad()  # 梯度置0
            real_out = D(real_img)  # 真实图片输入判别器
            Dloss_real = criterion(real_out, real_label)  # 计算loss
            real_scores = real_out  
            fake_img = G(fake).detach()    # 使用detach避免梯度传到G，因为G不用更新
            fake_out = D(fake_img) 
            Dloss_fake = criterion(fake_out, fake_label)  
            fake_scores = fake_out  
            Dloss = (Dloss_real + Dloss_fake) 
            Dloss.backward()  # 反向传播
            Doptimizer.step()  # 更新参数
            
            #生成器训练
            fake = Variable(torch.randn(num, fake_dimension)).cuda()  #再次生成随机输入，否则生成器不易收敛
            Goptimizer.zero_grad()  
            fake_img = G(fake)  
            out = D(fake_img)
            Gloss = criterion(out, real_label)  
            Gloss.backward()  
            Goptimizer.step()  

    
            if (i % 100 == 0): #输出中间结果
                print('Epoch[{}/{}],Batch{},D_loss:{:.7f},G_loss:{:.7f} '
                    'D real: {:.7f},D fake: {:.7f}'.format(
                    epoch,  num_epoch, i,Dloss.data.item(), Gloss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean()  
                ))
        if epoch == 0:
            real_images = convert2img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')  # 保存真实图片
        if(epoch<100 or epoch%100==0):
            if(epoch%200==0):# and epoch>0):
                torch.save(G.state_dict(), './generator{}.pth'.format(epoch)) # 保存模型
                torch.save(D.state_dict(), './discriminator{}.pth'.format(epoch))
            fake_images = convert2img(fake_img.cpu().data)
            save_image(fake_images, './img/fake_images-{}.png'.format(epoch)) # 保存图片

 
def test(D,G): #测试
    for i, (img, _) in enumerate(testloader):
        num = img.size(0)
        img = img.view(num,-1)
        real_image = Variable(img).cuda()
        fake = Variable(torch.randn(num, fake_dimension)).cuda()
        
        fake_image = G(fake)       
        real_out = D(real_image)
        fake_out = D(fake_image)
        
        if(i%50==0):
            fake_images = convert2img(fake_image.cpu().data) # 保存图片
            save_image(fake_images, './testres/batch-{}.png'.format(i))
        print('test batch:{}  D fake test: {}  D real test: {}'.format(i,fake_out.data.mean(),real_out.data.mean()))

gpath = './generator500.pth'
dpath = './discriminator500.pth'

if __name__=='__main__':
    D = discriminator()
    G = generator() 
    D = D.to(DEVICE)
    G = G.to(DEVICE) 
    train(D,G)
    d1 = discriminator()
    g1 = generator()
    d1.load_state_dict(torch.load(dpath))
    g1.load_state_dict(torch.load(gpath)) #加载保存的模型
    d1 = d1.to(DEVICE)
    g1 = g1.to(DEVICE)
    test(d1,g1)
 

