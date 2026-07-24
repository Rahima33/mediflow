"""
Grad-CAM implementation for MediFlow's DenseNet121 pneumonia classifier.

Grad-CAM (Gradient-weighted Class Activation Mapping) answers:
"Which regions of this X-ray most influenced the model's prediction?"

It works by hooking into the last convolutional layer of the network,
capturing both its output (activations) during the forward pass and
its gradients during the backward pass, then combining them into a
spatial heatmap.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms


class GradCAM:
    """
    Grad-CAM for a DenseNet121 binary classifier.

    Hooks into a target convolutional layer to capture:
      - activations: what patterns the layer detected (forward pass)
      - gradients: how much each of those patterns influenced the
        predicted class score (backward pass)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        # Forward hook: runs automatically whenever target_layer processes
        # a forward pass. We grab its output, and register a second,
        # tensor-level hook on that same output to capture its gradient
        # during the backward pass. This avoids register_full_backward_hook,
        # which conflicts with DenseNet's in-place ReLU immediately downstream.
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Run Grad-CAM on a single preprocessed image tensor.

        Args:
            input_tensor: shape [1, 3, 224, 224], already normalized
            class_idx: which class to explain (0=NORMAL, 1=PNEUMONIA).
                       If None, uses the model's own top prediction.

        Returns:
            cam: 2D numpy array (values 0-1), the raw heatmap
            class_idx: the class that was explained
            confidence: softmax probability for that class
        """
        # ---- Forward pass ----
        output = self.model(input_tensor)  # shape: [1, 2]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # ---- Backward pass from the predicted class's score ----
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # ---- Channel-wise importance weights ----
        # self.gradients shape: [1, 1024, 7, 7] for DenseNet121
        # Average over the spatial dimensions -> one importance
        # weight per feature-map channel.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, 1024, 1, 1]

        # ---- Weighted combination of feature maps ----
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = F.relu(cam)  # keep only regions with a positive influence on this class

        # ---- Normalize to [0, 1] for visualization ----
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        confidence = output.softmax(dim=1)[0, class_idx].item()
        return cam, class_idx, confidence


def create_model():
    """Rebuild the DenseNet121 architecture used during training."""
    model = models.densenet121(weights=None)  # we load our own fine-tuned weights
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    return model


def load_trained_model(checkpoint_path, device="cpu"):
    """Load the fine-tuned MediFlow checkpoint into a fresh DenseNet121."""
    model = create_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load an X-ray and prepare it exactly like during training/evaluation."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension -> [1, 3, 224, 224]
    return image, tensor


def overlay_heatmap(original_image, cam, alpha=0.4):
    """Blend the Grad-CAM heatmap on top of the original X-ray."""
    original_np = np.array(original_image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (heatmap * alpha + original_np * (1 - alpha)).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DEVICE = "cpu"
    CHECKPOINT_PATH = "models/best_model_truecrop.pth"
    IMAGE_PATH = "sample_images/NORMAL2-IM-0092-0001.jpeg"  # swap in a real test image
    CLASSES = ["NORMAL", "PNEUMONIA"]

    model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)

    # DenseNet121's last convolutional layer (still has spatial structure,
    # right before global pooling + the classifier head flattens it away)
    target_layer = model.features.norm5

    gradcam = GradCAM(model, target_layer)

    original_image, input_tensor = preprocess_image(IMAGE_PATH)
    cam, predicted_class, confidence = gradcam.generate(input_tensor)

    print(f"Predicted: {CLASSES[predicted_class]} ({confidence:.2%} confidence)")

    overlay = overlay_heatmap(original_image, cam)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {CLASSES[predicted_class]}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()