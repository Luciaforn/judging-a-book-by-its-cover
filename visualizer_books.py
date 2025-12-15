import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image


class BookModelVisualizer:
    def __init__(self, model, device, class_names):
        """
        Utility to visualize predictions, Grad-CAM and metrics
        of the model on book covers.

        Args:
            model: PyTorch model already loaded (same architecture used in training).
            device: torch.device('cuda') or torch.device('cpu').
            class_names: list of strings with the class names
                         in the same order used by train_val_dataset.classes.
        """
        self.model = model
        self.device = device
        self.class_names = class_names

        # Mean / std also used in the notebook for Normalize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    # ------------------------------------------------------------------
    #  Image denormalization
    # ------------------------------------------------------------------
    def denormalize(self, tensor):
        """
        Converts a normalized tensor (C, H, W) into a NumPy image (H, W, C)
        with values in [0,1], ready for plt.imshow.
        """
        img = tensor.cpu().numpy()          # (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        img = img * self.std + self.mean    # denormalization
        img = np.clip(img, 0, 1)
        return img

    # ------------------------------------------------------------------
    #  Visualization of Top-3 predictions on random images
    # ------------------------------------------------------------------
    def plot_predictions(self, dataset, num_samples=5):
        """
        Selects some random images from the dataset,
        shows the covers, the true label and the model's Top-3 predictions.

        dataset: typically val_dataset or test_dataset
        num_samples: how many images to show (maximum 5 for layout reasons)
        """
        self.model.eval()

        # 🔧 Limit to at most 5 images, as requested
        n_samples = min(num_samples, len(dataset), 5)

        # Random indices
        indices = random.sample(range(len(dataset)), n_samples)

        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 5))
        if n_samples == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            image, label_idx = dataset[idx]  # image: tensor (C,H,W), label_idx: int
            ax = axes[i]

            # Prepare input
            input_tensor = image.unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)        # logits [1, num_classes]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, 3, dim=1)

            preds = top_indices[0].cpu().numpy()
            probs_np = top_probs[0].cpu().numpy()

            img_np = self.denormalize(image)
            ax.imshow(img_np)
            ax.axis("off")

            # True and predicted labels
            true_label_name = self.class_names[label_idx]
            top1_idx = int(preds[0])
            top1_name = self.class_names[top1_idx]

            # Evaluation logic
            if top1_idx == label_idx:
                status = "TOP-1"
                status_color = "green"
            elif int(label_idx) in preds:
                status = "TOP-3"
                status_color = "orange"
            else:
                status = "FAIL"
                status_color = "red"

            title = f"True: {true_label_name}\nPred: {top1_name}\n{status}"
            ax.set_title(title, color=status_color, fontsize=12, fontweight="bold")

            # Add probabilities as text below the image
            info_text = ""
            for j in range(3):
                cls_name = self.class_names[int(preds[j])]
                prob_val = probs_np[j] * 100
                marker = "-" if int(preds[j]) == int(label_idx) else ""
                info_text += f"{j+1}. {cls_name}: {prob_val:.1f}% {marker}\n"

            ax.text(
                0.5, -0.1, info_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
            )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Single example (without Grad-CAM)
    # ------------------------------------------------------------------
    def visualize_single_prediction(self, image_tensor, true_label_idx=None):
        """
        Visualizes a single image (normalized) with the Top-3 predictions.

        image_tensor: tensor (C,H,W) already normalized (same transforms as the dataset).
        true_label_idx: index of the true label (optional).
        """
        self.model.eval()

        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, 3, dim=1)

        preds = top_indices[0].cpu().numpy()
        probs_np = top_probs[0].cpu().numpy()

        img_np = self.denormalize(image_tensor)

        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.axis("off")

        # Title: true vs top-1 prediction
        if true_label_idx is not None:
            true_name = self.class_names[true_label_idx]
        else:
            true_name = "Unknown"

        top1_idx = int(preds[0])
        top1_name = self.class_names[top1_idx]

        if true_label_idx is None:
            status = "Pred only"
            status_color = "black"
        elif top1_idx == true_label_idx:
            status = "TOP-1 "
            status_color = "green"
        elif int(true_label_idx) in preds:
            status = "TOP-3 "
            status_color = "orange"
        else:
            status = "FAIL "
            status_color = "red"

        title = f"True: {true_name}\nPred: {top1_name}\n{status}"
        plt.title(title, color=status_color, fontsize=12, fontweight="bold")

        # Top-3 in a textbox at the bottom
        info_text = ""
        for j in range(3):
            cls_name = self.class_names[int(preds[j])]
            prob_val = probs_np[j] * 100
            marker = ""
            if true_label_idx is not None and int(preds[j]) == int(true_label_idx):
                marker = "-"
            info_text += f"{j+1}. {cls_name}: {prob_val:.1f}% {marker}\n"

        plt.gcf().text(
            0.5, 0.02, info_text,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Grad-CAM: generate heatmap for a single image
    # ------------------------------------------------------------------
    def generate_gradcam(self, input_tensor, target_class=None, target_layer_name="layer4"):
        """
        Computes Grad-CAM for a single image (C,H,W) already normalized (ImageNet).

        Args:
            input_tensor: image tensor (C,H,W) normalized.
            target_class: index of the target class; if None uses the predicted class.
            target_layer_name: name of the convolutional layer to compute Grad-CAM on
                               (for ResNet50 typically 'layer4').

        Returns:
            cam: 2D numpy array in [0,1] with shape [H, W]
            target_class: integer index of the target class used for the backward.
        """
        self.model.eval()

        # [1, C, H, W]
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations["value"] = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        target_layer = dict(self.model.named_modules())[target_layer_name]
        handle_fwd = target_layer.register_forward_hook(fwd_hook)

        if hasattr(target_layer, "register_full_backward_hook"):
            handle_bwd = target_layer.register_full_backward_hook(bwd_hook)
        else:
            handle_bwd = target_layer.register_backward_hook(bwd_hook)

        output = self.model(input_tensor)  # [1, num_classes]

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        acts = activations["value"]   # [1, C, H', W']
        grads = gradients["value"]    # [1, C, H', W']

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()  # [H, W]

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, target_class

    # ------------------------------------------------------------------
    #  Grad-CAM: visualization of image + heatmap
    # ------------------------------------------------------------------
    def show_gradcam_on_image(self, input_tensor, cam,
                              pred_class_idx=None, true_label_idx=None, example_idx=None):
        """
        input_tensor: single image tensor (C,H,W), normalized.
        cam: 2D numpy array [H,W] in [0,1].
        pred_class_idx: predicted class index (for the title of the right panel).
        true_label_idx: true label index (for the left panel).
        example_idx: running index for labeling in the title.
        """
        img = self.denormalize(input_tensor)  # (H,W,C)

        plt.figure(figsize=(6, 3))

        # Original image + true label
        plt.subplot(1, 2, 1)
        title_left = "Original"
        if example_idx is not None:
            title_left += f" #{example_idx}"
        if true_label_idx is not None:
            try:
                title_left += f"\nTrue: {self.class_names[true_label_idx]}"
            except Exception:
                title_left += f"\nTrue idx: {true_label_idx}"
        plt.title(title_left)
        plt.imshow(img)
        plt.axis("off")

        # Image + heatmap
        plt.subplot(1, 2, 2)
        title = "Grad-CAM"
        if pred_class_idx is not None:
            try:
                title += f"\nPred: {self.class_names[pred_class_idx]}"
            except Exception:
                title += f"\nPred idx: {pred_class_idx}"
        plt.title(title)
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=0.4)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Grad-CAM: multi-image examples on dataset (val/test)
    # ------------------------------------------------------------------
    def plot_gradcam_examples(self, dataset, num_examples=5, seed=None, target_layer_name="layer4"):
        """
        Selects some random images from the dataset (val/test),
        computes Grad-CAM and shows image + heatmap for each.

        Args:
            dataset: typically val_dataset or test_dataset (same transforms as training).
            num_examples: how many images to show.
            seed: if specified (int), makes the choice of images reproducible.
            target_layer_name: layer on which to compute Grad-CAM (default: 'layer4').
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        num_to_show = min(num_examples, len(dataset))
        indices = random.sample(range(len(dataset)), num_to_show)

        for example_idx, idx in enumerate(indices, start=1):
            sample_img, sample_label = dataset[idx]  # image already transformed (resize + normalize)
            cam, pred_cls = self.generate_gradcam(sample_img, target_layer_name=target_layer_name)

            print(f"\n=== Example {example_idx} (dataset index {idx}) ===")
            try:
                true_name = self.class_names[sample_label]
            except Exception:
                true_name = f"idx {sample_label}"
            try:
                pred_name = self.class_names[pred_cls]
            except Exception:
                pred_name = f"idx {pred_cls}"

            print(f"True label:      {sample_label} ({true_name})")
            print(f"Predicted class: {pred_cls} ({pred_name})")

            self.show_gradcam_on_image(
                sample_img,
                cam,
                pred_class_idx=pred_cls,
                true_label_idx=sample_label,
                example_idx=example_idx
            )

      # ------------------------------------------------------------------
    #  Visualization of data augmentation (original vs transformed)
    # ------------------------------------------------------------------
    def show_augmentation_example(
        self,
        dataset,
        transform,
        num_examples=1,
        seed=None,
        title_prefix="Data augmentation"
    ):
        """
        Shows, for some images taken from the dataset (using the CSV and root_dir),
        the comparison between:
          - original image (file on disk)
          - image after the same transforms used in training/validation.

        Args:
            dataset: instance of BookCoverDataset (with attributes .df and .root_dir).
            transform: torchvision.transforms pipeline (e.g. data_transforms['train']).
            num_examples: how many original/transformed pairs to show.
            seed: optional, to make the image selection reproducible.
            title_prefix: prefix for the title (e.g. 'Train Augmentation').
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        num_to_show = min(num_examples, len(dataset))
        indices = random.sample(range(len(dataset)), num_to_show)

        for i, idx in enumerate(indices, start=1):
            # Retrieve image path from CSV
            try:
                row = dataset.df.iloc[idx]
                filename = str(row["Filename"])
                img_path = os.path.join(dataset.root_dir, filename)
            except Exception as e:
                print(f" Unable to get path for idx={idx}: {e}")
                continue

            # Load original image
            try:
                orig_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f" Unable to open {img_path}: {e}")
                continue

            # Apply the same transform pipeline (ToTensor + Normalize, etc.)
            transformed_tensor = transform(orig_img)  # (C,H,W)

            # Convert for display
            orig_np = np.array(orig_img) / 255.0  # [H,W,3] in [0,1]
            aug_np = self.denormalize(transformed_tensor)

            plt.figure(figsize=(8, 4))

            # Original
            plt.subplot(1, 2, 1)
            plt.imshow(orig_np)
            plt.title(f"{title_prefix} - Original #{i}")
            plt.axis("off")

            # Transformed
            plt.subplot(1, 2, 2)
            plt.imshow(aug_np)
            plt.title(f"{title_prefix} - Transformed #{i}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    #  Confusion Matrix (val or test)
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, dataloader, normalize=False,
                              title="Matrice di Confusione - Generi Letterari"):
        """
        Computes and visualizes the confusion matrix on a dataloader
        (typically validation or test).

        Args:
            dataloader: DataLoader (e.g. test_loader).
            normalize: if True, normalizes per row (percentages per class).
            title: plot title.
        """
        self.model.eval()
        y_true = []
        y_pred = []

        print(" Calcolo delle predizioni per la Matrice di Confusione...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(20, 16))  # Large size to fit 30 classes
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        plt.ylabel('Vero Genere (True Label)', fontsize=14)
        plt.xlabel('Genere Predetto (Predicted Label)', fontsize=14)
        plt.title(title, fontsize=18)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return cm

    # ------------------------------------------------------------------
    #  Loss curves (train vs validation)
    # ------------------------------------------------------------------
    def plot_loss_curves(self, history, title="Training vs Validation Loss"):
        """
        Plots training/validation loss curves over time.

        Args:
            history: dict with keys 'train_loss' and 'val_loss' (as built in train_model).
            title: plot title.
        """
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])

        if not train_loss or not val_loss:
            print(" history does not contain 'train_loss' and/or 'val_loss'. Nothing to plot.")
            return

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
