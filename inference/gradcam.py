import os

import cv2
import numpy as np
import torch
import torch.nn as nn


def _find_last_conv_layer(model):
    last_conv = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No convolution layer found in DenseNet features.")
    return last_conv


def _build_input_tensor(image):
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor.requires_grad_(True)
    return tensor


def _to_uint8_rgb(image_rgb):
    image = np.asarray(image_rgb)
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    min_v = float(np.min(image))
    max_v = float(np.max(image))
    if max_v > min_v:
        image = (image - min_v) / (max_v - min_v)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def generate_gradcam_overlay(model, image_rgb, output_path, focus_index=None):
    """
    Generate Grad-CAM heatmap from DenseNet's final convolutional features.
    """
    model.eval()

    activations = {}
    gradients = {}

    target_conv = _find_last_conv_layer(model)

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = target_conv.register_forward_hook(forward_hook)
    bh = target_conv.register_full_backward_hook(backward_hook)

    try:
        img_tensor = _build_input_tensor(image_rgb)

        logits = model(img_tensor)
        if focus_index is None:
            focus_index = int(torch.argmax(logits, dim=1).item())
        focus_score = logits[0, focus_index]

        model.zero_grad(set_to_none=True)
        focus_score.backward()

        grads = gradients["value"].detach().cpu().numpy()[0]
        acts = activations["value"].detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_rgb_uint8 = _to_uint8_rgb(image_rgb)
        original_bgr = cv2.cvtColor(original_rgb_uint8, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(heatmap, 0.4, original_bgr, 0.9, 0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, superimposed)
        return output_path
    finally:
        fh.remove()
        bh.remove()