# scalable-image-classification-cifar100

Deep Learning project: CNN + ResNet50 fine-tuning on CIFAR-100 with performance optimization and evaluation.

This project fine-tunes a pre-trained ResNet-50 model on the CIFAR-100 dataset using PyTorch. 
It was completed as a capstone project and follows a two-stage fine-tuning approach: 
head training followed by full fine-tuning of deeper layers.
   

## Contents
 
   - `notebook.ipynb` — Full training and evaluation pipeline


## Dataset

- **Dataset:** CIFAR-100
- **Total training datapoints:** 50,000 images
- **Number of classes:** 100
- **Input size:** Images resized to 224×224 to match ResNet-50 input requirements

CIFAR-100 Dataset: https://huggingface.co/datasets/uoft-cs/cifar100


## Model

- **Base model:** ResNet-50 (pre-trained on ImageNet)
- **Pre-processing transforms:**
  - Resize: 232px → Center crop: 224px
  - Normalisation mean: [0.485, 0.456, 0.406]
  - Normalisation std: [0.229, 0.224, 0.225]
  - Interpolation: Bilinear


## Training

### Stage 1 — Head Training (10 Epochs)
Only the final fully connected layer (`fc.weight`, `fc.bias`) was trained while all other layers were frozen.

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|-----------|----------------|----------|--------------|
| 1     | 3.4944    | 20.69%         | 3.0218   | 28.96%       |
| 5     | 2.5401    | 36.87%         | 2.5799   | 36.28%       |
| 10    | 2.3650    | 40.54%         | 2.4704   | 38.41%       |

### Stage 2 — Full Fine-Tuning (35 Epochs)
Deeper layers were unfrozen and the entire model was fine-tuned end-to-end.

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|-----------|----------------|----------|--------------|
| 1     | 2.1568    | 45.39%         | 2.2114   | 43.66%       |
| 10    | 1.5956    | 57.74%         | 1.7478   | 54.06%       |
| 20    | 1.4210    | 61.56%         | 1.5835   | 57.57%       |
| 35    | 1.2460    | 65.82%         | 1.4849   | 59.98%       |


## Results

| Metric | Value |
|--------|-------|
| Final Training Accuracy | **65.82%** |
| Final Validation Accuracy | **59.98%** |
| Final Training Loss | 1.2460 |
| Final Validation Loss | 1.4849 |


## Sample Prediction

The fine-tuned model was tested on a sample image:

- **Input:** `apple_s_000022.png`
- **Predicted label:** `apple` ✅


## Environment

- Platform: Google Colab (GPU: CUDA)
- Framework: PyTorch
- Model saved as: `resnet50_cifar100_finetuned.pth`


## Files

| File | Description |
|------|-------------|
| `notebook.ipynb` | Full training pipeline — data loading, model setup, training, evaluation and prediction |
| `resnet50_cifar100_finetuned.pth` | Saved fine-tuned model weights |


## What I Would Do Differently

I would experiment with a learning rate scheduler to reduce the learning rate as 
training progressed, which likely would have squeezed out a few more percentage 
points of accuracy.


> **Note:** Model weights (`resnet50_cifar100_finetuned.pth`) are not included 
> due to file size. Re-run the notebook to reproduce them.
