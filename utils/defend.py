from .defended_vlm import VLM, DefendedVLM
from .classifier import ImageClassifier, DefendedImageClassifier
from .transfer_attack import attack_step_pgd
from .utils import print_memory_consumption, get_memory_consumption
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision import transforms as T
from datasets import Dataset
import random

@dataclass
class DefenceConfig:
    model_name_clf: str
    model_name_vlm: str
    n_gradient_steps: int
    user_query: str
    lr_vlm: float
    lr_clf: float
    max_perturbation_pixels_vlm: int
    max_perturbation_pixels_clf: int
    use_integrated_gradients: bool
    ig_n: int = 5
    print_every: int = 2


class LTNet(nn.Module):
    def __init__(self, in_feats=4):
        super(LTNet, self).__init__()
        #self.conv1=torch.nn.Conv1d(in_channels=in_feats, out_channels=12, kernel_size=2, stride=1)
        #self.avgpool1 = nn.AdaptiveAvgPool1d(12)
        #self.norm1=nn.BatchNorm1d(12)
        #self.conv2=torch.nn.Conv1d(in_channels=12, out_channels=12, kernel_size=2, stride=1)
        #self.avgpool2 = nn.AdaptiveAvgPool1d(12)
        #self.norm2=nn.BatchNorm1d(12)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

        self.linear1 = nn.Linear(3*224*224,100)# 9*224*224)
        #self.linear2 = nn.Linear(12*12*4, 12*12*4)
        self.fc = nn.Linear(100, 3*224*224)
        self.flatten=nn.Flatten()#torch.flatten


    def forward(self, x):
        """
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.relu(x)
        x = self.norm2(x)
        """
        x = self.flatten(x)#, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.linear2(x)
        #x = self.relu(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x.view(-1,3,224,224)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, loss1, loss2, grader):
        grads_clf = torch.autograd.grad(loss1, grader, create_graph=True, allow_unused=False)[0]
        grads_vlm = torch.autograd.grad(loss2, grader, create_graph=True, allow_unused=False)[0]
        grad_similarity_loss = torch.nn.functional.cosine_similarity(grads_clf.flatten(), grads_vlm.flatten(), dim=0)
        return grad_similarity_loss


def defend_using_prompt(
        prompt_image: torch.tensor,
        additive_image: torch.tensor,
        vlm: DefendedVLM,
        clf: DefendedImageClassifier,
        ds: Dataset,
        idx2label: dict,
        config: DefenceConfig,
        device: str,
        mask_flag: bool,
        grad_only: bool,
        idx=38,
        adv_in=None,
):

    n_gradient_steps = config.n_gradient_steps
    user_query = config.user_query
    max_perturbation_pixels_vlm = config.max_perturbation_pixels_vlm
    max_perturbation_pixels_clf = config.max_perturbation_pixels_clf
    lr_vlm = config.lr_vlm
    lr_clf = config.lr_clf
    use_integrated_gradients = config.use_integrated_gradients
    ig_n = config.ig_n
    print_every = config.print_every

    prompt_image = prompt_image.float()
    prompt_image.requires_grad = True
    prompt_image_initial = prompt_image.clone().float() if device=="cuda" else prompt_image.clone()

    additive_image = additive_image.float()
    #print([additive_image.min(), additive_image.max()])
    additive_image=clf.preprocess(additive_image)['pixel_values']
    #input([additive_image.min(), additive_image.max()])
    #additive_image.requires_grad = True
    additive_image_initial = additive_image.clone().float() if device=="cuda" else additive_image.clone()
    best_loss=100000
    sampled_idx = idx
    for i in range(n_gradient_steps):
        # ensure that image requires grad
        if not grad_only:
            prompt_image.requires_grad = True
            additive_image.requires_grad = True



            row = ds[sampled_idx]
            sampled_image = row['image']
            sampled_label = row['label']
            sampled_label_str = idx2label[sampled_label].split(", ")[0]

            sampled_image = T.PILToTensor()(sampled_image)
        else:
            sampled_image=adv_in

            sampled_image_lon = T.Resize((224,224))(sampled_image).float()
            sampled_image_lon.requires_grad = True
            #print(sampled_image)
            #input(sampled_image_lon)
        if not grad_only:
            sampled_image=clf.preprocess(sampled_image_lon.clone())
            #input(sampled_image['pixel_values'])
            #input(torch.max(sampled_image['pixel_values']))
            sampled_image['pixel_values'].requires_grad=True
        else:
            sampled_image={'pixel_values':sampled_image}
            sampled_image['pixel_values'].requires_grad=True
        #sampled_image.retain_grad()
        logits, sampled_label, sampled_label_str = clf.classify(sampled_image, additive_image=additive_image)
        #print([laby, labys])
        #print([sampled_label, sampled_label_str])
        #input()

        full_text_vlm_prompt, target_tokens = vlm.get_training_prompt(user_query, sampled_label_str)


        loss_clf = clf.compute_clf_loss(logits, sampled_label)

        out = vlm.forward(sampled_image_lon, full_text_vlm_prompt, prompt_image=prompt_image, overwrite=True)
        loss_vlm = vlm.compute_gen_loss(out, target_tokens)

        #print(loss_clf)
        #input(loss_vlm)
        #logits.sum()
        #tzt=sampled_image.sum()
        #print(tzt)
        #input(loss_clf)
        #loss_clf.backward()
        if not use_integrated_gradients:
            grads_clf = torch.autograd.grad(loss_clf, sampled_image['pixel_values'], create_graph=True, allow_unused=False)[0]
            #input(grads_clf)
            grads_vlm = torch.autograd.grad(loss_vlm, sampled_image_lon, create_graph=True, allow_unused=False)[0]
            #input(grads_vlm)


        else:
            mock_images = [prompt_image, sampled_image]
            grads_clf = compute_integrated_gradient(clf, sampled_image, mock_images, full_text_vlm_prompt, target_tokens, sampled_label, prompt_image, additive_image, n=ig_n)[0]
            grads_vlm = compute_integrated_gradient(vlm, sampled_image, mock_images, full_text_vlm_prompt, target_tokens, sampled_label, prompt_image, additive_image, n=ig_n)[0]

        #print(grads_vlm.shape)

        #input(grads_clf.shape)
        if mask_flag:
            with torch.no_grad():
                mask=((grads_clf.flatten() > 0) * (grads_vlm.flatten() > 0)) > 0

            grad_similarity_loss = torch.nn.functional.cosine_similarity(grads_clf.flatten()[mask], grads_vlm.flatten()[mask], dim=0)
        else:
            #grad_similarity_loss = torch.nn.functional.cosine_similarity(grads_clf.flatten(), grads_vlm.flatten(), dim=0)
            grad_similarity_loss = -torch.nn.functional.mse_loss(grads_clf.flatten(), grads_vlm.flatten())
        # loss_vlm_mod = loss_vlm if loss_vlm > 10 else 0
        #loss_vlm_mod = 0
        #print([prompt_image.min(), prompt_image.max(), additive_image.min(), additive_image.max()])
        #input()
        #input(grad_similarity_loss)
        if grad_only:
            return grad_similarity_loss.detach()
        prompt_grad, additive_grad = torch.autograd.grad(grad_similarity_loss, [prompt_image, additive_image], create_graph=False, allow_unused=False)

        if (i+1) % print_every == 0:
            print(f"Iter {i+1:4d}/{n_gradient_steps} >>>  RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Classifier: {loss_clf.item():.6f}, VLM: {loss_vlm.item():.6f}, Gradient Similarity: {grad_similarity_loss.item():.9f}")

        # optimization step
        with torch.no_grad():
            prompt_image = attack_step_pgd(prompt_image, prompt_grad, lr_vlm, max_perturbation_pixels_vlm, prompt_image_initial)
            additive_image = attack_step_pgd(additive_image, additive_grad, lr_clf, max_perturbation_pixels_clf, additive_image_initial, additive=True)

        if grad_similarity_loss < best_loss:
            best_prompt_image=prompt_image.clone()
            best_additive_image=additive_image.clone()

    return best_prompt_image, best_additive_image



def forward_pass(
        model,
        input_image: torch.tensor,
        mock_images: list,
        full_text_vlm_prompt: str,
        target_tokens,
        sampled_label: int,
        prompt_image: torch.tensor = None,
        additive_image: torch.tensor = None,
    ):
    if isinstance(model, DefendedVLM):
        out = model.forward(input_image, mock_images, full_text_vlm_prompt, prompt_image=prompt_image, overwrite=True)
        loss = model.compute_gen_loss(out, target_tokens)
    elif isinstance(model, VLM):
        out = model.forward(input_image, mock_images, full_text_vlm_prompt, overwrite=True)
        loss = model.compute_gen_loss(out, target_tokens)
    elif isinstance(model, DefendedImageClassifier):
        logits, _, _, = model.classify(input_image, additive_image=additive_image)
        loss = model.compute_clf_loss(logits, sampled_label)
    elif isinstance(model, ImageClassifier):
        logits, _, _, = model.classify(input_image)
        loss = model.compute_clf_loss(logits, sampled_label)

    return loss

def defend_using_net(
        prompt_image: torch.tensor,
        vlm: DefendedVLM,
        clf: DefendedImageClassifier,
        ds: Dataset,
        idx2label: dict,
        config: DefenceConfig,
        device: str,
        mask_flag: bool,
):

    n_gradient_steps = config.n_gradient_steps
    user_query = config.user_query
    max_perturbation_pixels_vlm = config.max_perturbation_pixels_vlm
    max_perturbation_pixels_clf = config.max_perturbation_pixels_clf
    lr_vlm = config.lr_vlm
    lr_clf = config.lr_clf
    use_integrated_gradients = config.use_integrated_gradients
    ig_n = config.ig_n
    print_every = config.print_every

    prompt_image = prompt_image.float()
    best_loss=100000
    input("wild")
    ltnet=LTNet().to('cuda')
    #ltnet=torch.nn.DataParallel(ltnet)

    ltneta=LTNet().to('cuda')
    #ltneta=torch.nn.DataParallel(ltneta)
    #ltnet.parameters.requires_grad=True
    #ltneta.parameters.requires_grad=True
    input("time time")
    print(ltnet)
    for layer in ltnet.children():
        if isinstance(layer, nn.Linear):
            print(layer.weight)
    #criterion = nn.CosineEmbeddingLoss()
    #optimizer_a = optim.SGD(ltnet.parameters(), lr=0.01, momentum=0.9)
    #optimizer_b = optim.SGD(ltneta.parameters(), lr=0.01, momentum=0.9)
    for i in range(n_gradient_steps):
        # ensure that image requires grad
        #optimizer_a.zero_grad()
        #optimzier_b.zero_grad()

        sampled_idx = 38
        row = ds[sampled_idx]
        sampled_image = row['image']
        sampled_label = row['label']
        sampled_label_str = idx2label[sampled_label].split(", ")[0]

        sampled_image = T.PILToTensor()(sampled_image)
        sampled_image_lon = T.Resize((224,224))(sampled_image).float()

        sampled_image=clf.preprocess(sampled_image_lon.clone())

        #sampled_image['pixel_values'].requires_grad=True

        full_text_vlm_prompt, target_tokens = vlm.get_training_prompt(user_query, sampled_label_str)
        input("go for it")
        #input(sampled_image.shape)
        logits, _, _, = clf.classify(sampled_image, lt=True, ltnet=ltneta)
        loss_clf = clf.compute_clf_loss(logits, sampled_label)
        #print(sampled_image.shape())
        #print([sampled_image.min(),sampled_image.max()])
        print(sampled_image_lon.shape)
        input([sampled_image_lon.min(),sampled_image_lon.max()])
        out = vlm.forward(sampled_image_lon, full_text_vlm_prompt, prompt_image=prompt_image, overwrite=True)
        loss_vlm = vlm.compute_gen_loss(out, target_tokens)

        #print(loss_clf)
        #input(loss_vlm)
        #logits.sum()
        #tzt=sampled_image.sum()
        #print(tzt)
        #input(loss_clf)
        #loss_clf.backward()
        input()
        if not use_integrated_gradients:
            grads_clf = torch.autograd.grad(loss_clf, ltneta.parameters(), create_graph=True, allow_unused=False)[0]
            input(grads_clf)
            grads_vlm = torch.autograd.grad(loss_vlm, ltnet.parameters(), create_graph=True, allow_unused=False)[0]
            input(grads_vlm)


        else:
            mock_images = [prompt_image, sampled_image]
            grads_clf = compute_integrated_gradient(clf, sampled_image, mock_images, full_text_vlm_prompt, target_tokens, sampled_label, prompt_image, additive_image, n=ig_n)[0]
            grads_vlm = compute_integrated_gradient(vlm, sampled_image, mock_images, full_text_vlm_prompt, target_tokens, sampled_label, prompt_image, additive_image, n=ig_n)[0]

        #print(grads_vlm.shape)

        #input(grads_clf.shape)
        if mask_flag:
            with torch.no_grad():
                mask=((grads_clf.flatten() > 0) * (grads_vlm.flatten() > 0)) > 0

            grad_similarity_loss = torch.nn.functional.cosine_similarity(grads_clf.flatten()[mask], grads_vlm.flatten()[mask], dim=0)
        else:
            grad_similarity_loss = torch.nn.functional.cosine_similarity(grads_clf.flatten(), grads_vlm.flatten(), dim=0)
        #grad_similarity_loss = 1-torch.nn.functional.mse_loss(grads_clf.flatten(), grads_vlm.flatten())
        # loss_vlm_mod = loss_vlm if loss_vlm > 10 else 0
        #loss_vlm_mod = 0
        #print([prompt_image.min(), prompt_image.max(), additive_image.min(), additive_image.max()])
        #input()
        prompt_grad, additive_grad = torch.autograd.grad(grad_similarity_loss, [prompt_image, additive_image], create_graph=False, allow_unused=False)

        if (i+1) % print_every == 0:
            print(f"Iter {i+1:4d}/{n_gradient_steps} >>>  RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Classifier: {loss_clf.item():.6f}, VLM: {loss_vlm.item():.6f}, Gradient Similarity: {grad_similarity_loss.item():.9f}")

        # optimization step
        with torch.no_grad():
            prompt_image = attack_step_pgd(prompt_image, prompt_grad, lr_vlm, max_perturbation_pixels_vlm, prompt_image_initial)
            additive_image = attack_step_pgd(additive_image, additive_grad, lr_clf, max_perturbation_pixels_clf, additive_image_initial, additive=True)

        if grad_similarity_loss < best_loss:
            best_prompt_image=prompt_image.clone()
            best_additive_image=additive_image.clone()

    return best_prompt_image, best_additive_image



def compute_integrated_gradient(
        model,
        input_image,
        mock_images: list,
        full_text_vlm_prompt: str,
        target_tokens,
        sampled_label: int,
        prompt_image: torch.tensor = None,
        additive_image: torch.tensor = None,
        baseline=None,
        n=20,
        learn_transformation=False,
    ):

    if baseline is None: baseline = torch.zeros_like(input_image)

    mean_grad = 0

    for i in range(1, n + 1):
        with torch.no_grad():
            x = baseline + i / n * (input_image - baseline)
        x.requires_grad = True
        loss = forward_pass(
            model,
            x,
            mock_images,
            full_text_vlm_prompt,
            target_tokens,
            sampled_label,
            prompt_image,
            additive_image,
        )
        (grad,) = torch.autograd.grad(loss, x)
        mean_grad += grad / n

    integrated_gradients = (input_image - baseline) * mean_grad

    return integrated_gradients, mean_grad
