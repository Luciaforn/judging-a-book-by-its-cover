import torch
import matplotlib.pyplot as plt
import numpy as np
import random

class BookModelVisualizer:
    def __init__(self, model, device, class_names):
        """
        Args:
            model: Il modello PyTorch caricato.
            device: 'cuda' o 'cpu'.
            class_names: Lista con i nomi delle classi (es. ['Arts', 'Biographies'...]).
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def denormalize(self, tensor):
        """Riporta l'immagine dai valori normalizzati di ImageNet ai colori reali per il plot."""
        img = tensor.cpu().numpy().transpose((1, 2, 0)) # Da (C,H,W) a (H,W,C)
        img = self.std * img + self.mean
        img = np.clip(img, 0, 1) # Assicura che i valori siano tra 0 e 1
        return img

    def plot_predictions(self, dataset, num_samples=5):
        """
        Prende N immagini random dal dataset, fa la predizione e mostra:
        - Immagine
        - Label Reale
        - Predizione (con indicazione se Top-1, Top-3 o Fail)
        """
        self.model.eval()
        
        # Indici random
        indices = random.sample(range(len(dataset)), num_samples)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 6))
        
        for i, idx in enumerate(indices):
            image, label_idx = dataset[idx]
            ax = axes[i]
            
            # Preparazione input
            input_tensor = image.unsqueeze(0).to(self.device)
            
            # Inferenza
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Prendiamo le top 3 probabilità e indici
                probs, preds = torch.topk(outputs, 3, dim=1)
                
            preds = preds.cpu().numpy()[0] # Array di 3 indici
            probs = torch.nn.functional.softmax(probs, dim=1).cpu().numpy()[0] # Probabilità
            
            real_label = self.class_names[label_idx]
            pred_top1 = self.class_names[preds[0]]
            
            # LOGICA DI VALUTAZIONE
            status_color = 'red'
            status_text = "FAIL ❌"
            
            if preds[0] == label_idx:
                status_color = 'green'
                status_text = "TOP-1 ✅"
            elif label_idx in preds:
                status_color = 'orange'
                status_text = "TOP-3 ⚠️"
                
            # Visualizzazione Immagine
            ax.imshow(self.denormalize(image))
            ax.axis('off')
            
            # Costruzione Titolo
            title = f"Real: {real_label}\n"
            title += f"Pred: {pred_top1}\n"
            title += f"({status_text})"
            
            ax.set_title(title, color=status_color, fontsize=12, fontweight='bold')
            
            # Aggiungiamo le probabilità sotto come testo
            info_text = ""
            for j in range(3):
                cls_name = self.class_names[preds[j]]
                prob_val = probs[j] * 100
                marker = "👈" if preds[j] == label_idx else ""
                info_text += f"{j+1}. {cls_name}: {prob_val:.1f}% {marker}\n"
            
            ax.text(0.5, -0.1, info_text, transform=ax.transAxes, 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        plt.tight_layout()
        plt.show()