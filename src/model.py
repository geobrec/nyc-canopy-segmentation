"""Small U-Net. Kept tiny on purpose - this trains on CPU."""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )
class SmallUNet(nn.Module):
    def __init__(self, in_ch=4, base=16):
        """in_ch=4 for NAIP R G B NIR. ~0.5M params at base=16."""
        super().__init__()
        self.e1 = block(in_ch, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.bott = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = block(base * 8, base * 4)   # 8x in, not 4x - the skip cat doubles it
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))

        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))

        return self.out(d1)   # logits - BCEWithLogitsLoss does the sigmoid
    