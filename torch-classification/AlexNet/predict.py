"""
#-*-coding:utf-8-*- 
# @anthor: wangyu a beginner programmer, striving to be the strongest.
# @date: 2021/7/21 16:48
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_path = "../../dataset/Bijie-landslide-dataset/images/landslide/df018.png"
    image = Image.open(image_path)
    plt.imshow(image)

    image = data_transform(image)
    image = torch.unsqueeze(image, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file {} does not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_dict = json.load(json_file)

    model = AlexNet(num_classes=2).to(device)
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(image.to(device)))
        predict = torch.softmax(output, dim=0).cpu()
        predict_class = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_dict[str(predict_class)],
                                                 predict[predict_class].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
