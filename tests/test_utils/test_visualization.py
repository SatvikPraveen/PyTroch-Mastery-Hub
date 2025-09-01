# tests/test_utils/test_visualization.py
"""
Tests for visualization utilities
"""

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.visualization import (
    plot_training_curves, plot_tensor_as_image, plot_gradient_flow,
    plot_confusion_matrix, visualize_model_architecture
)


class TestPlotTrainingCurves:
    """Test training curve plotting."""
    
    def test_plot_loss_only(self):
        """Test plotting loss curves only."""
        train_losses = [0.9, 0.7, 0.5, 0.3, 0.2]
        val_losses = [0.8, 0.6, 0.5, 0.4, 0.3]
        
        fig = plot_training_curves(train_losses, val_losses)
        
        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)
    
    def test_plot_with_accuracy(self):
        """Test plotting curves with accuracy."""
        train_losses = [0.9, 0.7, 0.5, 0.3, 0.2]
        val_losses = [0.8, 0.6, 0.5, 0.4, 0.3]
        train_accs = [0.6, 0.7, 0.8, 0.85, 0.9]
        val_accs = [0.55, 0.65, 0.75, 0.8, 0.85]
        
        fig = plot_training_curves(
            train_losses, val_losses, train_accs, val_accs
        )
        
        assert fig is not None
        assert len(fig.get_axes()) == 2
        plt.close(fig)
    
    def test_plot_train_only(self):
        """Test plotting training curves only."""
        train_losses = [0.9, 0.7, 0.5, 0.3, 0.2]
        
        fig = plot_training_curves(train_losses)
        
        assert fig is not None
        plt.close(fig)


class TestPlotTensorAsImage:
    """Test tensor image plotting."""
    
    def test_plot_2d_tensor(self):
        """Test plotting 2D tensor as image."""
        tensor = torch.randn(32, 32)
        
        fig = plot_tensor_as_image(tensor)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_3d_tensor(self, sample_image_tensor):
        """Test plotting 3D tensor as image."""
        fig = plot_tensor_as_image(sample_image_tensor)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_normalization(self):
        """Test plotting with normalization."""
        tensor = torch.randn(32, 32) * 255  # Large values
        
        fig = plot_tensor_as_image(tensor, normalize=True)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_numpy_array(self):
        """Test plotting numpy array."""
        array = np.random.randn(32, 32)
        
        fig = plot_tensor_as_image(array)
        
        assert fig is not None
        plt.close(fig)


class TestPlotGradientFlow:
    """Test gradient flow plotting."""
    
    def test_gradient_flow_plot(self, sample_model):
        """Test plotting gradient flow."""
        # Create dummy gradients
        for param in sample_model.parameters():
            param.grad = torch.randn_like(param) * 0.01
        
        fig = plot_gradient_flow(sample_model.named_parameters())
        
        assert fig is not None
        plt.close(fig)
    
    def test_gradient_flow_no_gradients(self, sample_model):
        """Test plotting when no gradients exist."""
        # Ensure no gradients
        for param in sample_model.parameters():
            param.grad = None
        
        fig = plot_gradient_flow(sample_model.named_parameters())
        
        assert fig is not None
        plt.close(fig)


class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""
    
    def test_confusion_matrix_basic(self):
        """Test basic confusion matrix plotting."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        fig = plot_confusion_matrix(y_true, y_pred)
        
        assert fig is not None
        plt.close(fig)
    
    def test_confusion_matrix_with_labels(self):
        """Test confusion matrix with class labels."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        class_names = ['Class A', 'Class B', 'Class C']
        
        fig = plot_confusion_matrix(
            y_true, y_pred, class_names=class_names
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        fig = plot_confusion_matrix(
            y_true, y_pred, normalize=True
        )
        
        assert fig is not None
        plt.close(fig)


class TestVisualizeModelArchitecture:
    """Test model architecture visualization."""
    
    def test_model_summary_no_torchsummary(self, sample_model):
        """Test model summary without torchsummary package."""
        # This should not raise an error
        visualize_model_architecture(sample_model, (10,))
    
    @pytest.mark.skipif(
        not pytest.importorskip("torchsummary", minversion=None),
        reason="torchsummary not available"
    )
    def test_model_summary_with_torchsummary(self, sample_model):
        """Test model summary with torchsummary package."""
        visualize_model_architecture(sample_model, (10,))