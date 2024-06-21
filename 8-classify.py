import torch
from torchvision import models
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os


class ViolenceClass:
    def __init__(self):
        #weight_path = './training_set_modified.ckpt' #'./best-acc.pth'
        weight_path = './test_model.pth'
        state_dict = torch.load(weight_path)        
        state_dict_to_load = self.remove_prefix_from_keys(state_dict['state_dict'], 'model.')
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.load_state_dict(state_dict_to_load)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.label = []
    
    #NOTE: the keys of the dict contain 'model.' prefix, which should be removed
    def remove_prefix_from_keys(self, original_dict, prefix):
        new_dict = {}
        prefix_length = len(prefix)
        
        for key, value in original_dict.items():
            if key.startswith(prefix):
                new_key = key[prefix_length:]
            else:
                new_key = key
            
            new_dict[new_key] = value
        return new_dict
    
    def transfer_imgs_to_tensor(self, imgs : list) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),          
        ])
        img_tensors = []
        for img in imgs:
            img_tensor = transform(img)
            img_tensors.append(img_tensor)
        batch_tensor = torch.stack(img_tensors)
        return batch_tensor
    
    def load_imgs_from_path(self, path) -> list:
        img_list = []
        for filename in os.listdir(path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.label.append(int(filename[0]))
                image_path = os.path.join(path, filename)
                image = Image.open(image_path).convert('RGB')
                img_list.append(image)
        return img_list
                        
    def classify(self, imgs : torch.Tensor):
        # 图像分类
        res = []
        probes = []
        dataset = TensorDataset(imgs)
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            with torch.no_grad():
                input_tensor = batch[0]
                input_tensor = input_tensor.to(self.device)
                outputs = self.model(input_tensor)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                outputs = outputs.cpu().numpy()
                for item in preds:
                    res.append(item)
                for item in outputs:
                    probes.append(item)
        return res, probes
    
    def accuracy_score(self, true_label : list , pred_label : list):
        cnt = 0
        for i in range(len(true_label)):
            if true_label[i] == pred_label[i]:
                cnt += 1
        return cnt / len(true_label)

    def make_predictions(self):
        img_path = "./test"
        imgs = self.load_imgs_from_path(path=img_path)
        imgs = self.transfer_imgs_to_tensor(imgs=imgs)
        preds, probes = self.classify(imgs=imgs)
        acc = self.accuracy_score(self.label, preds)
        print(acc)
        return self.label, preds, probes

if __name__ == "__main__":
    pred = ViolenceClass()
    pred.make_predictions()
