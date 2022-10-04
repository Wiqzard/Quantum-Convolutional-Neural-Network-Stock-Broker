import torch
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from networks import *
from data_processing import *


best_model_path = "/content/best_train_experiment1.pth"
model = Net().to(device)
model.load_state_dict(torch.load(best_model_path))
model.eval()
loss_func = nn.CrossEntropyLoss()

with torch.no_grad():
    total_loss = []
    correct = 0
    for data, target in test_dataloader:
        data = data.float().cuda()
        target = target.float().cuda()
        output = model(data).cuda()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print(
        "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss),
            correct / len(test_dataloader) * 100 / batch_size,
        )
    )


predictions = []
true = []
for square, label in test_dataloader:
    predictions.append(
        model(square.float().to(device)).argmax(dim=1, keepdim=True).item()
    )
    true.append(label.argmax(dim=1, keepdim=True).item())

f1_weighted = sklearn.metrics.f1_score(
    true, predictions, labels=None, average="weighted", sample_weight=None
)
print("F1 score (weighted)", f1_weighted)
print(
    "F1 score (micro)",
    sklearn.metrics.f1_score(
        true, predictions, labels=None, average="micro", sample_weight=None
    ),
)
print("cohen's Kappa", sklearn.metrics.cohen_kappa_score(true, predictions))

conf_mat = sklearn.metrics.confusion_matrix(true, predictions)
print(conf_mat)


recall = []
for i, row in enumerate(conf_mat):
    recall.append(np.round(row[i] / np.sum(row), 2))
    print(f"Recall of class {i} = {recall[i]}")
print("Recall avg", sum(recall) / len(recall))
