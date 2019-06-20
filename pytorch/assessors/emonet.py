import torchvision
import torch
import math

class EmoNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.model.fc= torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]

def emonet(tencrop):
    model = EmoNet()
    parameters = torch.load("./assessors/EmoNet_valence_moments_resnet50_5_best.pth.tar", map_location='cpu')
    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(parameters['state_dict'].items())}
    state_dict["model.fc.weight"] = state_dict.pop("model.last_linear.weight")
    state_dict["model.fc.bias"] = state_dict.pop("model.last_linear.bias")
    model.load_state_dict(state_dict)

    if tencrop:
        input_transform = tencrop_image_transform(model)
        output_transform = tencrop_output_transform_emonet
    else:
        input_transform = image_transform(model)
        output_transform = lambda x: x

    return model, input_transform, output_transform

def tencrop_image_transform(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: tencrop(image.permute(0, 2, 3, 1), cropped_size=224)),
        torchvision.transforms.Lambda(lambda image: torch.stack([torch.stack([normalize(x / 255) for x in crop])
                                                                 for crop in image])),
    ])

def tencrop_output_transform_emonet(output):
    output = output.view(-1, 10).mean(1)
    return output

def image_transform(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
        torchvision.transforms.Lambda(lambda image: torch.stack([normalize(x / 255) for x in image])),
    ])


def tencrop(images, cropped_size=227):
    im_size = 256  # hard coded

    crops = torch.zeros(images.shape[0], 10, 3, cropped_size, cropped_size)
    indices = [0, im_size - cropped_size]  # image size - crop size

    for img_index in range(images.shape[0]):  # looping over the batch dimension
        img = images[img_index, :, :, :]
        curr = 0
        for i in indices:
            for j in indices:
                temp_img = img[i:i + cropped_size, j:j + cropped_size, :]
                crops[img_index, curr, :, :, :] = temp_img.permute(2, 0, 1)
                crops[img_index, curr + 5, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
                curr = curr + 1
        center = int(math.floor(indices[1] / 2) + 1)
        crops[img_index, 4, :, :, :] = img[center:center + cropped_size,
                                           center:center + cropped_size, :].permute(2, 0, 1)
        crops[img_index, 9, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
    return crops
