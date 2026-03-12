from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification
import torch
from strenum import StrEnum

class ClassifierName(StrEnum):
    RESNET_50 = "microsoft/resnet-50"
    VIT_B_P16_224 = "google/vit-base-patch16-224"
    VIT_L_P16_224 = "google/vit-large-patch16-224"

VIT_MODELS = [
    ClassifierName.VIT_B_P16_224,
    ClassifierName.VIT_L_P16_224
]

ALL_CLASSIFIERS = [
    ClassifierName.RESNET_50,
    ClassifierName.VIT_B_P16_224,
    ClassifierName.VIT_L_P16_224,
]

class ImageClassifier:


    def __init__(self, model_name: str, device: str):
        self.name = model_name
        self.device = device

        if model_name == ClassifierName.RESNET_50:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.model = ResNetForImageClassification.from_pretrained(model_name).eval().to(device)

        if model_name in VIT_MODELS:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                attn_implementation="eager"# if device=="mps" else "auto"
            ).to(device)


    def classify(self, image):
        #self.processor(image, return_tensors="pt")
        return self._classify(image)

    def _classify(self, inputs):
        """
        Outputs class index and logits
        """
        #input("may I?")

        inputs = self.processor(inputs, return_tensors="pt").to(self.device)
        #input(inputs.keys())
        #return image.sum(), 0, 0
        #return inputs['pixel_values'].sum(), 0, 0
        #input(inputs)
        #INVESTIGATE: FOR CWA MODEL WAS ON CPU... HAS THIS BEEN THE CASE ALL ALONG???
        ##print(inputs)
        #input(self.model.device)

        logits = self.model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        predicted_label_str = self.model.config.id2label[predicted_label]

        return logits, predicted_label, predicted_label_str
    def forward_logits(self, image):
        logits, _, _ = self.classify(image)
        return logits

    def compute_clf_loss(self, logits, target_label_idx):
        num_classes = logits.shape[-1]
        onehot_target = torch.nn.functional.one_hot(torch.tensor(target_label_idx), num_classes=num_classes).unsqueeze(0).float().to(self.device)
        return torch.nn.CrossEntropyLoss()(logits, onehot_target)


class DefendedImageClassifier(ImageClassifier):

    additive_image = None
    ltnet=None
    def classify(self, image, additive_image=None, lt=False, ltnet=None):
        if additive_image is None:
            additive_image = self.additive_image
        if ltnet is None:
            ltnet = self.ltnet

        #image = self.add_image(image, additive_image)
        if not lt:
            image['pixel_values']=torch.clamp(image['pixel_values']+additive_image, min=-1, max=1)
            return self._classify(image)
        else:
            input(image['pixel_values'].shape)
            image['pixel_values']=ltnet(image['pixel_values']).cpu()
            input(image['pixel_values'].shape)
            return self._classify(image)


    def add_image(self, image: torch.tensor, additive_image: torch.tensor):
        return torch.clamp(image + additive_image, 0, 255)

    def preprocess(self, image):
        return self.processor(image, return_tensors="pt")
