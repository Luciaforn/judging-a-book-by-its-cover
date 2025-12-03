# Book Cover Genre Classifier – Demo Web App

This is a small Tornado web app that uses a trained ResNet-50 model to classify book cover images into literature genres and show the Top 1 / Top 2 / Top 3 predictions with a simple UI.

## Requirements

- Python 3.9+ (tested with Python 3.10)
- The following Python packages:

```bash
pip install torch torchvision tornado pillow
```
- Get the resnet50_bookcovers.pth  trained model running the notebook until the model is trained

## Installation
- Put resnet50_bookcovers.pth in the project directory
- Run demo.py
- You should see this message: "Server running on http://localhost:8080"
- In your browser you should see:
![enter image description here](https://imgur.com/a/cubnOUy)
- enjoy :)
