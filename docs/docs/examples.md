# docs/docs/examples.md

# Examples & Tutorials

Explore practical examples and hands-on tutorials to master the ML Training Framework.

## 🎯 Quick Examples

### Image Classification in 10 Lines

```python
from src.computer_vision.models import SimpleResNet
from src.neural_networks.training import Trainer
from src.utils.data_utils import get_cifar10_loaders

# Load data and create model
train_loader, val_loader = get_cifar10_loaders(batch_size=32)
model = SimpleResNet(num_classes=10)

# Train
trainer = Trainer(model, train_loader, val_loader, device='cuda')
trainer.fit(epochs=10)

# Results automatically logged to MLflow!
```

### Text Classification with Transformers

```python
from src.nlp.models import TransformerClassifier
from src.nlp.tokenization import SimpleTokenizer

# Setup
tokenizer = SimpleTokenizer(vocab_size=10000)
model = TransformerClassifier(vocab_size=10000, num_classes=2)

# Train on your text data
trainer = Trainer(model, train_loader, val_loader)
trainer.fit(epochs=5)
```

### GAN Training

```python
from src.advanced.gan_utils import DCGAN, GANTrainer

# Create and train GAN
gan = DCGAN(noise_dim=100, img_channels=3)
trainer = GANTrainer(gan.generator, gan.discriminator)
trainer.train(data_loader, epochs=50)
```

## 📚 Interactive Notebooks

### 🔰 Fundamentals (Start Here!)

#### **[01. Introduction to Tensors](../notebooks/01_fundamentals/01_introduction_to_tensors.ipynb)**

Learn PyTorch tensors, operations, and GPU acceleration basics.

```python
# What you'll learn:
- Tensor creation and manipulation
- Broadcasting and indexing
- GPU acceleration
- Memory management
```

#### **[02. Gradient Computation](../notebooks/01_fundamentals/02_gradient_computation.ipynb)**

Understanding automatic differentiation and gradients.

```python
# Key concepts:
- Autograd system
- Gradient computation
- Chain rule implementation
- Custom backward functions
```

#### **[03. Custom Autograd Functions](../notebooks/01_fundamentals/03_custom_autograd_functions.ipynb)**

Build your own differentiable operations.

#### **[04. Backpropagation Visualization](../notebooks/01_fundamentals/04_backpropagation_visualization.ipynb)**

Visualize how gradients flow through networks.

### 🧠 Neural Networks

#### **[05. MLP from Scratch](../notebooks/02_neural_networks/05_mlp_from_scratch.ipynb)**

Build a multi-layer perceptron without high-level APIs.

```python
# Build your own:
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

#### **[06. Advanced Architectures](../notebooks/02_neural_networks/06_advanced_architectures.ipynb)**

Explore ResNets, attention mechanisms, and modern architectures.

#### **[07. Training Techniques](../notebooks/02_neural_networks/07_training_techniques.ipynb)**

Master advanced training strategies and optimization.

### 👁️ Computer Vision

#### **[08. CNN Fundamentals](../notebooks/03_computer_vision/08_cnn_fundamentals.ipynb)**

Convolutional networks for image processing.

```python
# Learn about:
- Convolution operations
- Pooling layers
- Feature maps
- Architecture design
```

#### **[09. Modern CNN Architectures](../notebooks/03_computer_vision/09_modern_cnn_architectures.ipynb)**

ResNet, EfficientNet, Vision Transformers.

#### **[10. Computer Vision Projects](../notebooks/03_computer_vision/10_computer_vision_projects.ipynb)**

End-to-end image classification project.

### 🔤 Natural Language Processing

#### **[11. RNN/LSTM Fundamentals](../notebooks/04_natural_language_processing/11_rnn_lstm_fundamentals.ipynb)**

Sequential models for text processing.

#### **[12. Sequence to Sequence](../notebooks/04_natural_language_processing/12_sequence_to_sequence.ipynb)**

Translation and text generation models.

#### **[13. Sentiment Analysis Project](../notebooks/04_natural_language_processing/13_sentiment_analysis_project.ipynb)**

Complete NLP project with preprocessing and deployment.

#### **[14. Transformer from Scratch](../notebooks/04_natural_language_processing/14_transformer_from_scratch.ipynb)**

Build the attention mechanism and transformer architecture.

### 🎨 Generative Models

#### **[15. GAN Fundamentals](../notebooks/05_generative_models/15_gan_fundamentals.ipynb)**

Generate realistic images with adversarial training.

```python
# GAN training loop:
for epoch in range(num_epochs):
    # Train discriminator
    d_loss = train_discriminator(real_data, fake_data)

    # Train generator
    g_loss = train_generator(noise)

    # Log results
    log_training_progress(d_loss, g_loss)
```

#### **[16. Advanced GANs & VAEs](../notebooks/05_generative_models/16_advanced_gans_vaes.ipynb)**

WGAN, StyleGAN, and Variational Autoencoders.

### 🚀 Optimization & Deployment

#### **[17. Model Optimization](../notebooks/06_optimization_deployment/17_model_optimization.ipynb)**

Quantization, pruning, and acceleration techniques.

#### **[18. Model Serving APIs](../notebooks/06_optimization_deployment/18_model_serving_apis.ipynb)**

Deploy models with REST APIs and containerization.

#### **[19. Monitoring & MLOps](../notebooks/06_optimization_deployment/19_monitoring_mlops.ipynb)**

Production monitoring, logging, and experiment tracking.

#### **[20. Cloud Deployment](../notebooks/06_optimization_deployment/20_cloud_deployment.ipynb)**

Deploy to AWS, GCP, and Kubernetes.

### 🏆 Advanced Projects

#### **[21. Image Classification Project](../notebooks/07_advanced_projects/21_image_classification_project.ipynb)**

End-to-end computer vision system.

#### **[22. Text Generation Project](../notebooks/07_advanced_projects/22_text_generation_project.ipynb)**

Build a language model for text generation.

#### **[23. Recommendation System](../notebooks/07_advanced_projects/23_recommendation_system.ipynb)**

Collaborative filtering and neural recommendations.

### 🔬 Advanced Topics

#### **[24. Advanced Techniques](../notebooks/08_advanced_topics/24_advanced_techniques.ipynb)**

Meta-learning, few-shot learning, and cutting-edge methods.

#### **[25. Research Applications](../notebooks/08_advanced_topics/25_research_applications.ipynb)**

Latest research implementations and techniques.

### 🎓 Capstone Projects

#### **[26. Multimodal System](../notebooks/capstone_projects/26_Capstone_part1_multimodal_system.ipynb)**

Combined vision and language processing system.

#### **[27. Production MLOps](../notebooks/capstone_projects/27_Capstone_part2_production_mlops.ipynb)**

Complete production deployment with monitoring.

## 🛠️ Code Examples by Task

### Data Loading & Preprocessing

```python
from src.utils.data_utils import create_dataloader
from src.computer_vision.transforms import ImageNetTransforms

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = load_data(data_path)
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Data loading
transform = ImageNetTransforms(size=224, mode='train')
loader = create_dataloader(MyDataset('data/', transform), batch_size=32)
```

### Custom Training Loop

```python
from src.neural_networks.training import Trainer

# Advanced trainer configuration
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer='adamw',
    scheduler='cosine',
    use_mixed_precision=True,
    accumulate_grad_batches=4,
    max_epochs=100,
    early_stopping_patience=10
)

# Custom callbacks
trainer.add_callback('model_checkpoint', save_top_k=3)
trainer.add_callback('lr_monitor')
trainer.add_callback('gpu_stats')

# Train with logging
trainer.fit()
```

### Model Evaluation

```python
from src.utils.metrics import ModelEvaluator, accuracy_score, f1_score

# Comprehensive evaluation
evaluator = ModelEvaluator(model)
results = evaluator.evaluate(test_loader, metrics=['accuracy', 'f1', 'precision', 'recall'])

# Custom metrics
def custom_metric(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

evaluator.add_metric('mse', custom_metric)
results = evaluator.evaluate(test_loader)
```

### Visualization

```python
from src.utils.visualization import plot_training_curves, visualize_predictions

# Training curves
plot_training_curves(
    train_losses=trainer.train_losses,
    val_losses=trainer.val_losses,
    metrics=['accuracy', 'f1_score']
)

# Prediction visualization
visualize_predictions(
    model=model,
    dataloader=test_loader,
    num_samples=16,
    save_path='predictions.png'
)
```

## 🎮 Interactive Examples

### Launch Examples Locally

```bash
# Start Jupyter Lab
jupyter lab notebooks/

# Or use Docker
docker-compose up jupyter
# Open http://localhost:8888
```

### Colab Integration

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-training/framework/blob/main/notebooks/)

Click any notebook to open directly in Google Colab!

### Binder Environment

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ml-training/framework/main?filepath=notebooks)

Run notebooks in a cloud environment without installation.

## 📖 Learning Paths

### 🎯 Beginner Path (2-3 weeks)

1. Fundamentals (notebooks 1-4)
2. Neural Networks (notebooks 5-7)
3. Choose: Computer Vision (8-10) OR NLP (11-13)
4. Basic project (21 or 22)

### 🚀 Intermediate Path (3-4 weeks)

1. Complete beginner path
2. Both Computer Vision AND NLP sections
3. Generative Models (15-16)
4. Optimization & Deployment (17-20)
5. Advanced project (23)

### 🔬 Advanced Path (4-6 weeks)

1. Complete intermediate path
2. Advanced Topics (24-25)
3. Capstone Projects (26-27)
4. Contribute to framework development

### 🏭 Production Path (2-3 weeks)

Focus on deployment and MLOps:

1. Model Optimization (17)
2. Model Serving APIs (18)
3. Monitoring & MLOps (19)
4. Cloud Deployment (20)
5. Production Capstone (27)

## 🤝 Community Examples

### Share Your Projects

Built something cool? Share it with the community!

- Submit examples via [Pull Request](https://github.com/ml-training/framework/pulls)
- Join our [Discord](https://discord.gg/ml-training) discussions
- Tag us on social media with `#MLTrainingFramework`

### Example Gallery

Check out community projects and implementations:

- **Image Segmentation** by @user1 - Medical image segmentation
- **Time Series Forecasting** by @user2 - Stock price prediction
- **Multi-modal Search** by @user3 - Text-image similarity search

---

**Ready to start?** Jump into the [fundamentals notebooks](../notebooks/01_fundamentals/) or try a [quick example](#quick-examples)!
