# Evaluation on CUB validation set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in cub_val_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on CUB validation images: {100 * correct / total:.2f}%')

## Accuracy on CUB validation images: 23.27% ##
