import asyncio
import io
import os
import urllib.request

import tornado.web

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


#configuration and model loading

# Path to the trained model (.pth)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet50_bookcovers.pth")

# Normalization used for ImageNet 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#classes list needed if the .pth doesn't already contain it
CLASSES = [
            "Arts & Photography",
            "Biographies & Memoirs",
            "Business & Money",
            "Calendars",
            "Children's Books",
            "Christian Books & Bibles",
            "Comics & Graphic Novels",
            "Computers & Technology",
            "Cookbooks, Food & Wine",
            "Crafts, Hobbies & Home",
            "Engineering & Transportation",
            "Health, Fitness & Dieting",
            "History",
            "Humor & Entertainment",
            "Law",
            "Literature & Fiction",
            "Medical Books",
            "Mystery, Thriller & Suspense",
            "Parenting & Relationships",
            "Politics & Social Sciences",
            "Reference",
            "Religion & Spirituality",
            "Romance",
            "Science & Math",
            "Science Fiction & Fantasy",
            "Self-Help",
            "Sports & Outdoors",
            "Teen & Young Adult",
            "Test Preparation",
            "Travel",
        ]
# Preprocessing pipeline for inference (resize -> tensor -> normalize)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# For portability CPU is forced (works on old GPUs / machines without CUDA)
#  to use a modern GPU, change this to:
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)


def build_model(num_classes: int) -> nn.Module:
    """
    Creates a ResNet-50 model pre-trained on ImageNet and replaces
    the final fully-connected layer so that it outputs `num_classes`
    genre logits.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def load_checkpoint(model_path: str, device: torch.device, classes:list[str]):
    """
    Loads the .pth checkpoint and returns:
      - model
      - classes (list of class names, index -> name)
      - class_to_idx (dict: name -> index)

    Supports two formats:
      1) checkpoint dict with keys, contains the weights and "classes" and "class_to_idx
      2) simple state_dict containing only weights, in this case, we recreate 'classes' and 'class_to_idx' locally.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    #Case 1 checkpoint containing classes and class_to_idx
    if isinstance(checkpoint, dict) and "classes" in checkpoint and "model_state_dict" in checkpoint:
        classes = checkpoint["classes"]
        class_to_idx = checkpoint["class_to_idx"]
        num_classes = len(classes)

        model = build_model(num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Case 2 base checkpoint with wegiths, classes and class_to_idx must be created
    else:
        #class list and its order, this order matches the sorted class list used during training.
        class_to_idx = {name: i for i, name in enumerate(classes)}
        num_classes = len(classes)

        model = build_model(num_classes)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, classes, class_to_idx



# The model is loaded once at the server startup
print(f"Loading model from: {MODEL_PATH}")
MODEL, CLASSES, CLASS_TO_IDX = load_checkpoint(MODEL_PATH, device,CLASSES)
print("Model loaded. Ready for inference.")


def predict_pil_image(img: Image.Image, topk: int = 3):
    """
    Run inference on a PIL image and return Top-k predictions as:
    [
        {"rank": 1, "class_name": "...", "prob": 0.85},
        {"rank": 2, "class_name": "...", "prob": 0.07},
        ...
    ]
    """
    img_t = preprocess(img)
    input_batch = img_t.unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = MODEL(input_batch)  
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_idxs = top_idxs.squeeze(0).cpu().numpy()

    predictions = []
    for rank, (cls_idx, prob) in enumerate(zip(top_idxs, top_probs), start=1):
        predictions.append(
            {
                "rank": int(rank),
                "class_name": CLASSES[int(cls_idx)],
                "prob": float(prob),
            }
        )
    return predictions


#Tornado Handlers


class MainHandler(tornado.web.RequestHandler):
    """Serves the main page """

    def get(self):
        self.render("index.html")


class ApiClassifyHandler(tornado.web.RequestHandler):
    """
    JSON API endpoint used by the front-end.

    Accepts either:
      - multipart/form-data with field "file" (uploaded image)
      - form field "image_url" (URL to an image).

    Returns:
      { "predictions": [ {rank, class_name, prob}, ... ] }
    """

    def post(self):
        image_url = self.get_argument("image_url", "").strip()
        files = self.request.files.get("file")
        img = None

        # Case 1 image URL
        if image_url:
            try:
                with urllib.request.urlopen(image_url) as response:
                    data = response.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                self.set_status(400)
                self.write({"error": "Could not download image from the given URL."})
                return

        # Case 2 uploaded file
        elif files:
            try:
                body = files[0]["body"]
                img = Image.open(io.BytesIO(body)).convert("RGB")
            except Exception:
                self.set_status(400)
                self.write({"error": "Uploaded file is not a valid image."})
                return

        # No input provided
        else:
            self.set_status(400)
            self.write({"error": "No file or URL provided."})
            return

        predictions = predict_pil_image(img, topk=3)
        self.set_header("Content-Type", "application/json")
        self.write({"predictions": predictions})


# App setup


def make_app():
    base_dir = os.path.dirname(__file__)
    settings = {
        "template_path": os.path.join(base_dir, "templates"),
        "static_path": os.path.join(base_dir, "static"),
        "debug": True,
    }
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/api/classify", ApiClassifyHandler),
        ],
        **settings,
    )


async def main():
    app = make_app()
    app.listen(8080)
    print("Server running on http://localhost:8080")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
