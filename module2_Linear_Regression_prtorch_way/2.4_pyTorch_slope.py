import torch
import matplotlib.pyplot as plt

print(torch.__version__)

# 1. The Hard way

w = torch.tensor(-1., requires_grad=True)

print("w is leaf?", w.is_leaf)   # w is leaf? True
X = torch.arange(-1, 3, 0.1).view(-1, 1) # view as only 1 column
f = -3*X

plt.plot(X, f)
plt.show()

Y = f + 0.1 * torch.rand(X.size())
plt.plot(X, Y,'.')
plt.show()

def forward(x):
    return w*X

def losskung(yhat, y):
    return torch.mean((yhat-y)**2)

lr = 0.1
COST = []
for epoch in range(4):
    # predict 
    yhat = forward(X)
    # print(yhat)

    # calculate loass and update weight
    loss = losskung(yhat, Y)
    loss = torch.mean((yhat-Y)**2)
    print(f"epoch #{epoch}, loss ", loss)
    loss.backward()  # calculate gradient
    print("w is leaf?", w.is_leaf)   # w is leaf? True

    # w = w - lr*w.grad   # บรรทัดนี้ทำให้ความเป็น leaf ของ w หายไป
    # print("w is leaf?", w.is_leaf)   # w is leaf? False

    w.data = w.data - lr*w.grad.data   # ใส่ .data จะเปลี่ยนแปลงแค่ค่าข้างใน tensor ไม่เปลี่ยนชนิด leaf
    print("w is leaf?", w.is_leaf)     # w is leaf? True

    # print(w.grad)
    w.grad.zero_()

    COST.append(loss.item())

plt.plot(range(len(COST)), COST)
plt.show()