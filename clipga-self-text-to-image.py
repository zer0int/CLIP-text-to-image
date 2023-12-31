
import os
import clip
import torch
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
from torch import nn

torch.autograd.set_detect_anomaly(True)
device="cuda"

def initialize_image(device):
    # Create a random noise image and directly set requires_grad to True
    image = torch.randn((1, 3, 224, 224), device=device) * 0.01
    image.requires_grad_(True)
    return image

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14', device)
    return model, preprocess, device

def encode_text(text, clip_model, preprocess, device):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    return text_features

def compute_similarity(image, text_features, clip_model, preprocess, device):
    # Remove batch dimension and convert tensor image to PIL Image for preprocessing
    pil_image = to_pil_image(image.squeeze(0).detach().clone().cpu())
    preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)

    image_features = clip_model.encode_image(preprocessed_image)

    # Cosine similarity as a loss
    loss = -torch.cosine_similarity(text_features, image_features).mean()
    return loss

def update_image(image, gradients, learning_rate):
    return image - learning_rate * gradients

def save_image(image, iteration, folder="images"):
    if iteration % 10 == 0:
        os.makedirs(folder, exist_ok=True)
        img = image.detach().clone().cpu().squeeze(0).permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        img = (img * 255).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"{folder}/image_{iteration}.png")

class MultiLayerActivationCaptureHook:
    def __init__(self, model):
        self.activations = {}
        for i, layer in enumerate(model.visual.transformer.resblocks):
            handle = layer.register_forward_hook(self.get_activation_hook(i))
            self.activations[i] = None

    def get_activation_hook(self, layer_idx):
        def hook(module, input, output):
            self.activations[layer_idx] = output.detach()
        return hook

def save_activations(activations, iteration, folder="activations"):
    if iteration % 10 == 0:
        os.makedirs(folder, exist_ok=True)
        for layer_idx, activation in activations.items():
            flat_act = activation.view(activation.size(0), -1)
            norm_act = (flat_act - flat_act.min()) / (flat_act.max() - flat_act.min())
            filename = f"{folder}/layer{layer_idx}_iteration{iteration}.npy"
            np.save(filename, norm_act.cpu().numpy())

def save_gradients(image, iteration, folder="gradients"):
    if iteration % 10 == 0:
        os.makedirs(folder, exist_ok=True)
        gradients = image.grad.view(image.grad.size(0), -1)
        norm_grad = (gradients - gradients.min()) / (gradients.max() - gradients.min())
        filename = f"{folder}/gradient_iteration{iteration}.npy"
        np.save(filename, norm_grad.cpu().numpy())

def total_variation_loss(img):
    """Calculate the Total Variation Loss."""
    tv_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
              torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return tv_loss

def gradient_ascent_loop(text, iterations, learning_rate, model, preprocess, device, softmax_temp=1.0, cls_weight=10.0, mem_lambda=0.01):
    image = initialize_image(device)
    text_features = encode_text(text, model, preprocess, device)
    activation_hook = MultiLayerActivationCaptureHook(model)
    optimizer = torch.optim.AdamW([image], lr=learning_rate)

    # Initialize memory variable to store the best state
    best_state = {
        'image': None,
        'similarity': float('-inf')
    }

    for iteration in range(iterations):
        optimizer.zero_grad()
        image = image.float()
        image_features = model.encode_image(image)

        # Calculate cosine similarity
        patch_cosine_sim = torch.cosine_similarity(image_features, text_features, dim=1)
        total_cosine_sim = patch_cosine_sim.mean()
        scaled_total_cosine_sim = torch.exp(total_cosine_sim / softmax_temp).mean()
        current_similarity = total_cosine_sim.item()

        # Update memory with the best state
        if current_similarity > best_state['similarity']:
            best_state['similarity'] = current_similarity
            best_state['image'] = image.detach().clone()

        # Calculate total variation loss
        tv_lambda = 0.00001
        tv_loss = total_variation_loss(image)
        
        # Calculate deviation from the best state (memory loss)
        if best_state['image'] is not None:
            deviation_loss = F.mse_loss(image, best_state['image'])
        else:
            deviation_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = -torch.log(scaled_total_cosine_sim) + tv_lambda * tv_loss + mem_lambda * deviation_loss

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Save activations, gradients, and image
        save_activations(activation_hook.activations, iteration)
        save_gradients(image, iteration)
        save_image(image, iteration)

        print(f"Iteration {iteration}, Total Cosine Similarity: {total_cosine_sim}, Loss: {loss}")

    return image

# Example usage
model, preprocess, device = load_clip_model()
# ❤️ is the text to optimize for, "create representations of <text>"
final_image = gradient_ascent_loop("❤️", 4000, 1.5, model, preprocess, device, softmax_temp=5)
