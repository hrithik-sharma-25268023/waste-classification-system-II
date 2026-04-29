# waste-classification-system-II

## Create the Environment

**conda env create --file environment.yml** 

#### Simple Explanation of All Models We Tried
This project tested many different image‑classification models to figure out which one works best for sorting waste into five categories: organic, cardboard, paper, plastic, textile.
Below is a simple explanation of each model, why we tried it, and what problems we faced.

# Very Small Images (32×32)
32×32×1 (Grayscale) – MobileNetV1 / V2 / V3
These images were too tiny and had almost no detail.

The models couldn’t see shapes or textures properly.

Result: accuracy was very low because the model basically saw blurry blobs.

# 32×32×3 (RGB) – MobileNetV1 / V2 / V3
Adding color helped a little, but the images were still too small.

The model could guess colors but not actual objects.

Result: still not usable for real waste classification.

## Small Images (48×48)
48×48×1 (Grayscale) – MobileNetV1 / V2 / V3
Without color, the model confused paper, plastic, and cardboard.

The resolution was still too low to see important details.

Result: accuracy improved slightly but still not good enough.

# 48×48×3 (RGB) – MobileNetV1 / V2 / V3
Better than grayscale, but still struggled with fine textures.

Some classes looked too similar at this size.

Result: okay performance, but not reliable.

## Medium Images (96×96)
(This turned out to be the sweet spot.)

# 96×96×3 – MobileNetV1
Simple and fast, worked decently.

Sometimes overfitted (memorized training images instead of learning).

Result: good baseline but not the best.

# 96×96×3 – MobileNetV2
Better at learning patterns and textures.

Needed class balancing to avoid bias.

Result: strong performance for a lightweight model.

# 96×96×3 – MobileNetV3 (Custom Tiny Version)
Added attention layers that helped the model focus on important parts of the image.

Very small model size, good accuracy.

Result: great balance between size and performance.

# 96×96×3 – MobileNetV3Small (Pretrained on ImageNet)
This was the best tiny model we tested.

Already trained on millions of images, so it learned our dataset quickly.

Result: high accuracy + very small file size → perfect for deployment.

## Large Images (128×128)
128×128×3 – MobileNetV1
More detail helped, but training became slower.

Still overfitted easily.

Result: decent but not worth the extra cost.

# 128×128×3 – MobileNetV2
Performed well with the extra resolution.

Needed more GPU memory and time.

Result: strong but heavier.

# 128×128×3 – MobileNetV3
Best accuracy overall because it could see more detail.

But also more expensive to train.

Result: great accuracy, but not ideal for small devices.

## EfficientNet Models
EfficientNetB0 (Pretrained)
Very accurate because it’s a powerful model.

But the file size was huge (around 16 MB).

Result: too big for mobile or embedded use.

## Custom Tiny Models
Tiny CNN (Sequential)
Extremely small and easy to understand.

But accuracy was limited because it was too simple.

Result: good for learning, not good for real‑world use.

## Tiny MobileNetV2‑Lite
Used depthwise layers to stay small but still learn patterns.

Better than the tiny CNN but still weaker than pretrained models.

Result: decent accuracy with very small size.

### Challenges We Faced During Model Development (Simple Explanation)
## Pretrained Models Were Too Big
We tried using powerful pretrained models like EfficientNet and MobileNetV3Large.
These models gave good accuracy, but the problem was:

They were too large in file size

They couldn’t fit into our deployment requirement (we needed under 400 KB)

They required more memory and were slower to run on small devices

So even though they performed well, we couldn’t use them because they didn’t meet our size limits.

## Custom CNN Models Were Too Weak
We also tried building our own tiny CNN models from scratch.
These were very small and easy to deploy, but:

They didn’t learn enough features

Accuracy was low

They kept failing to recognize similar classes

They struggled with real‑world variations like lighting, texture, and shape

Basically, they were too simple to understand the complexity of waste images.

## Finding the Right Balance Was Hard
We needed a model that was:

Small enough to deploy

Accurate enough to be useful

Fast enough to run on limited hardware

Most models were either:

1. Too big but accurate, or

2. Small but not accurate

This made the process challenging because we had to test many combinations of:

1. Image sizes

2. Model architectures

Pretrained vs. custom models

