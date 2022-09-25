import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch, io, json, os


class Prediction:
    def __init__(self):
        self.img_class_map = None
        self.mapping_file_path = 'index_to_name.json'
        self.model = models.densenet121(pretrained=True)
        self.model.eval()
        self.setImageClassMap()
    
    def setImageClassMap(self):
        if os.path.isfile(self.mapping_file_path):
            with open(self.mapping_file_path) as f:
                self.img_class_map = json.load(f)

    def transform_image(self, image):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
        return my_transforms(image).unsqueeze(0)

   

    def get_prediction(self, image):
        tensor = self.transform_image(image)
        predict = self.model(tensor)
        predict_class = torch.argmax(predict).item()
        return predict_class

    def render_prediction(self, prediction_idx):
        stridx = str(prediction_idx)
        class_name = 'Unknown'
        if self.img_class_map is not None:
            if stridx in self.img_class_map is not None:
                print('Hello world')
                class_name = self.img_class_map[stridx][1]
        return prediction_idx, class_name


