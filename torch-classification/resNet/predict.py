"""
#-*-coding:utf-8-*- 
# @anthor: wangyu a beginner programmer, striving to be the strongest.
# @date: 2021/7/20 14:50
"""
import json
import torch
from torchvision import transforms
from PIL import Image

from model import resnet50


def main():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    json_file = open("./class_indices.json", "r")
    class_dict = json.load(json_file)

    net = resnet50(num_classes=2)
    net.load_state_dict(torch.load('resNet50.pth'))

    img = Image.open('../../dataset/Aba_dataset/Aba_3/landslide_jpeg/140.jpeg')
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)

    net.eval()
    with torch.no_grad():
        outputs = torch.squeeze(net(img))
        predicts = torch.softmax(outputs, dim=0)
        print(outputs)
        print(predicts)
        predict = torch.argmax(predicts).numpy()
    print("class: {}, prob: {:.3}".format(class_dict[str(predict)],
                                          predicts[predict].numpy()))


if __name__ == '__main__':
    main()