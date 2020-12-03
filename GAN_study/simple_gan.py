# https://dreamgonfly.github.io/blog/gan-explained/ 블로그 참고

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


transform = transforms.Compose([
    transforms.ToTensor(), # data 를 파이토치의 tensor 형식으로 바꾼다. [rows][cols][channels] -> [channels][rows][cols] rows -> height cols -> width
    transforms.Normalize(mean=(0.5,), std=(0.5, ))
])

mnist = datasets.MNIST(root='data', download=True, transform=transform)

dataloader = DataLoader(mnist, batch_size=60, shuffle=True)

# 생성자는 랜덤 벡터 z를 입력으로 받아 가짜 이미지를 출력한다.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh())
    def forward(self, inputs):
        return self.main(inputs).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        return self.main(inputs)

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))


for epoch in range(100):
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)

        real_data = Variable(real_data)

        target_real = Variable(torch.ones(batch_size, 1))
        target_fake = Variable(torch.zeros(batch_size, 1))

        # 진짜 이미지 구분자에 넣기
        D_result_from_real = D(real_data)

        # 구분자의 출력값이 정답지인 1에서 멀수록 loss 가 높아짐.
        D_loss_real = criterion(D_result_from_real, target_real)

        # 생성자에 입력으로 줄 랜덤 백터 z
        z = Variable(torch.randn((batch_size, 100)))

        # 생성자로 가짜 이미지 생성
        fake_data = G(z)

        # 생성자가 만든 가짜 이미지를 구분자에 넣기
        D_result_from_fake = D(fake_data)

        # 구분자의 출력값이 정답지인 0에서 멀 수록 loss 가 높아진다.
        D_loss_fake = criterion(D_result_from_fake, target_fake)

        # 구분자의 loss 는 두 문제에서 계산된 loss 의 합
        D_loss = D_loss_real + D_loss_fake

        # 구분자의 매개 변수의 미분값을 0으로 초기화
        D.zero_grad()
        # 역전파를 이용해 매개 변수의 loss에 대한 미분값 계싼
        D_loss.backward()

        # 최적화 기법을 이용해 구분자의 매개 변수를 업데이트
        D_optimizer.step()

        # 생성자에 입력으로 줄 랜덤 벡터 z 만들기
        z = Variable(torch.randn((batch_size, 100)))
        z = z.cuda()

        fake_data = G(z)
        D_result_from_fake = D(fake_data)

        # 생성자의 입장에서 구분자의 출력값이 1에서 멀수록 loss 가 높아진다.
        G_loss = criterion(D_result_from_fake, target_real)

        G.zero_grad()

        G_loss.backward()
        G_optimizer.step()