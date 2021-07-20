"""
#-*-coding:utf-8-*- 
# @anthor: wangyu a beginner programmer, striving to be the strongest.
# @date: 2021/7/20 14:50
"""
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets

from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))

    transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "dataset", "Bijie-landslide-dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    print("图像路径：", image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=transform["train"])
    train_num = len(train_dataset)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=transform["val"])
    validate_num = len(validate_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, validate_num))

    # {'landslide': 0, 'non-landslide': 1}
    landslide_list = train_dataset.class_to_idx
    # print(landslide_list)
    class_dict = dict((val, key) for key, val in landslide_list.items())
    # print(class_dict)   # {0: 'landslide', 1: 'non-landslide'}
    json_str = json.dumps(class_dict, indent=1)
    # print(json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=nw)
    # val_data_iter = iter(validate_loader)
    # val_image, val_label = val_data_iter.next()

    net = resnet50()
    weight_path = "./resnet50-pre.pth"
    net.load_state_dict(torch.load(weight_path, map_location=device))
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 2)
    net.to(device)

    loss_func = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    EPOCHS = 3
    best_acc = 0.0
    save_path = './resNet50.pth'
    train_steps = len(train_loader)
    # print(len(train_loader)) # train_num / batch_size
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            output = net(images.to(device))
            loss = loss_func(output, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, EPOCHS, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predicts = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predicts, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, EPOCHS)

        val_accurate = acc / validate_num
        print("[epoch %d] train_loss: %.3f val_accuracy:%.3f" %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Training Finished!")


if __name__ == '__main__':
    main()
