class CUB_Dataset(Dataset):
    def __init__(self,img_file, label_file, transform=None):
        self.img =np.load(img_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image,label

cub_bird_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

cub_train_dataset = CUB_Dataset(img_file="./CUB_train_images.npy",
                                        label_file="./CUB_train_labels.npy",transform=cub_bird_transform)
cub_train_loader = torch.utils.data.DataLoader(cub_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

cub_val_dataset = CUB_Dataset(img_file="./CUB_val_images.npy",
                                        label_file="./CUB_val_labels.npy",transform=cub_bird_transform)
cub_val_loader = torch.utils.data.DataLoader(cub_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
