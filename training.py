import torch
from torchsummary import summary

from networks import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)


summary(model, (1, 15, 15))

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

epochs = 500
loss_list = []
# BUY => 1, SELL => 0, HOLD => 2
model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data = data.cuda()
        target = target.cuda()

        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print(
        "Training [{:.0f}%]\tLoss: {:.4f}".format(
            100.0 * (epoch + 1) / epochs, loss_list[-1]
        )
    )


path_to_save_model = "/quantum-machine-learning/saved_models/"
experiment = 1
torch.save(
    model.state_dict(), f"{path_to_save_model}best_train_experiment{experiment}.pth"
)
