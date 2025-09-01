# src/advanced/deployment.py
"""
Model serving utilities for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
import json
import time
from pathlib import Path
import logging


class ModelServer:
    """
    Simple model server for deployment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        
        self.model.eval()
        self.request_count = 0
        self.total_inference_time = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction results with metadata
        """
        start_time = time.time()
        
        try:
            # Preprocessing
            if self.preprocessing_fn:
                processed_input = self.preprocessing_fn(input_data)
            else:
                processed_input = input_data
            
            # Ensure tensor is on correct device
            if isinstance(processed_input, torch.Tensor):
                processed_input = processed_input.to(self.device)
            
            # Inference
            with torch.no_grad():
                raw_output = self.model(processed_input)
            
            # Postprocessing
            if self.postprocessing_fn:
                output = self.postprocessing_fn(raw_output)
            else:
                output = raw_output.cpu().numpy() if isinstance(raw_output, torch.Tensor) else raw_output
            
            inference_time = time.time() - start_time
            
            # Update statistics
            self.request_count += 1
            self.total_inference_time += inference_time
            
            result = {
                'prediction': output,
                'inference_time': inference_time,
                'request_id': self.request_count,
                'status': 'success'
            }
            
            self.logger.info(f"Prediction completed in {inference_time:.4f}s")
            
        except Exception as e:
            result = {
                'error': str(e),
                'inference_time': time.time() - start_time,
                'request_id': self.request_count + 1,
                'status': 'error'
            }
            
            self.logger.error(f"Prediction failed: {str(e)}")
        
        return result
    
    def batch_predict(self, input_batch: List[Any]) -> List[Dict[str, Any]]:
        """
        Make predictions on batch of inputs.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of prediction results
        """
        results = []
        
        # TODO: Implement true batch processing
        for input_data in input_batch:
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        avg_inference_time = (
            self.total_inference_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_inference_time,
            'throughput_rps': self.request_count / max(self.total_inference_time, 1e-6),
            'device': str(self.device)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'status': 'healthy',
            'model_loaded': True,
            'device': str(self.device),
            'stats': self.get_stats()
        }


class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def export_trace(
        self,
        example_input: torch.Tensor,
        filepath: str
    ) -> torch.jit.ScriptModule:
        """
        Export model using tracing.
        
        Args:
            example_input: Example input for tracing
            filepath: Output file path
            
        Returns:
            TorchScript module
        """
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(filepath)
        
        print(f"Model traced and saved to {filepath}")
        return traced_model
    
    def export_script(self, filepath: str) -> torch.jit.ScriptModule:
        """
        Export model using scripting.
        
        Args:
            filepath: Output file path
            
        Returns:
            TorchScript module
        """
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(filepath)
        
        print(f"Model scripted and saved to {filepath}")
        return scripted_model
    
    def optimize_for_inference(self, model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Optimize TorchScript model for inference.
        
        Args:
            model: TorchScript model
            
        Returns:
            Optimized model
        """
        optimized_model = torch.jit.optimize_for_inference(model)
        return optimized_model


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def export(
        self,
        example_input: torch.Tensor,
        filepath: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            example_input: Example input tensor
            filepath: Output ONNX file path
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration
            opset_version: ONNX opset version
        """
        torch.onnx.export(
            self.model,
            example_input,
            filepath,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
        
        print(f"Model exported to ONNX format: {filepath}")
    
    def validate_onnx(self, filepath: str, example_input: torch.Tensor) -> bool:
        """
        Validate exported ONNX model.
        
        Args:
            filepath: ONNX file path
            example_input: Example input for validation
            
        Returns:
            True if validation passes
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Load ONNX model
            onnx_model = onnx.load(filepath)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(filepath)
            
            # Get input name
            input_name = ort_session.get_inputs()[0].name
            
            # Run inference
            with torch.no_grad():
                pytorch_output = self.model(example_input)
            
            onnx_output = ort_session.run(None, {input_name: example_input.numpy()})
            
            # Compare outputs
            if torch.allclose(pytorch_output, torch.from_numpy(onnx_output[0]), atol=1e-6):
                print("ONNX validation passed!")
                return True
            else:
                print("ONNX validation failed: outputs don't match")
                return False
                
        except ImportError:
            print("ONNX validation requires 'onnx' and 'onnxruntime' packages")
            return False
        except Exception as e:
            print(f"ONNX validation failed: {str(e)}")
            return False


class TensorRTOptimizer:
    """
    Optimize models for TensorRT deployment.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def optimize_with_torch_tensorrt(
        self,
        example_input: torch.Tensor,
        precision: str = 'fp16'
    ) -> torch.jit.ScriptModule:
        """
        Optimize model using torch-TensorRT.
        
        Args:
            example_input: Example input tensor
            precision: Precision mode ('fp32', 'fp16', 'int8')
            
        Returns:
            TensorRT optimized model
        """
        try:
            import torch_tensorrt
            
            # Convert to TorchScript first
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Compile with TensorRT
            if precision == 'fp16':
                compiled_model = torch_tensorrt.compile(
                    traced_model,
                    inputs=[torch_tensorrt.Input(example_input.shape)],
                    enabled_precisions=torch.half
                )
            elif precision == 'int8':
                compiled_model = torch_tensorrt.compile(
                    traced_model,
                    inputs=[torch_tensorrt.Input(example_input.shape)],
                    enabled_precisions=torch.int8
                )
            else:  # fp32
                compiled_model = torch_tensorrt.compile(
                    traced_model,
                    inputs=[torch_tensorrt.Input(example_input.shape)]
                )
            
            print(f"Model optimized with TensorRT ({precision} precision)")
            return compiled_model
            
        except ImportError:
            print("TensorRT optimization requires 'torch-tensorrt' package")
            return None


def serve_model(
    model_path: str,
    host: str = '0.0.0.0',
    port: int = 8000,
    preprocessing_fn: Optional[Callable] = None,
    postprocessing_fn: Optional[Callable] = None
) -> None:
    """
    Serve model with simple HTTP API.
    
    Args:
        model_path: Path to saved model
        host: Server host
        port: Server port
        preprocessing_fn: Preprocessing function
        postprocessing_fn: Postprocessing function
    """
    try:
        from flask import Flask, request, jsonify
        
        # Load model
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            model = torch.load(model_path, map_location='cpu')
        else:
            model = torch.jit.load(model_path)
        
        # Create server
        server = ModelServer(model, preprocessing_fn=preprocessing_fn, postprocessing_fn=postprocessing_fn)
        
        # Create Flask app
        app = Flask(__name__)
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                result = server.predict(data['input'])
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify(server.health_check())
        
        @app.route('/stats', methods=['GET'])
        def stats():
            return jsonify(server.get_stats())
        
        print(f"Starting model server on {host}:{port}")
        app.run(host=host, port=port)
        
    except ImportError:
        print("Model serving requires 'flask' package")


def export_model(
    model: nn.Module,
    example_input: torch.Tensor,
    output_dir: str,
    formats: List[str] = ['torchscript', 'onnx']
) -> Dict[str, str]:
    """
    Export model to multiple formats.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor
        output_dir: Output directory
        formats: List of export formats
        
    Returns:
        Dictionary mapping format to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    exported_files = {}
    
    if 'torchscript' in formats:
        ts_exporter = TorchScriptExporter(model)
        ts_path = output_dir / 'model.pt'
        ts_exporter.export_trace(example_input, str(ts_path))
        exported_files['torchscript'] = str(ts_path)
    
    if 'onnx' in formats:
        onnx_exporter = ONNXExporter(model)
        onnx_path = output_dir / 'model.onnx'
        onnx_exporter.export(example_input, str(onnx_path))
        exported_files['onnx'] = str(onnx_path)
    
    # Save model metadata
    metadata = {
        'input_shape': list(example_input.shape),
        'model_type': model.__class__.__name__,
        'export_formats': formats,
        'exported_files': exported_files
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model exported to {len(formats)} formats in {output_dir}")
    return exported_files


class DeploymentConfig:
    """
    Configuration for model deployment.
    """
    
    def __init__(self):
        self.config = {
            'model_path': None,
            'device': 'cpu',
            'batch_size': 1,
            'max_batch_delay': 0.1,  # seconds
            'optimization': {
                'quantization': False,
                'pruning': False,
                'tensorrt': False
            },
            'monitoring': {
                'enable_metrics': True,
                'log_requests': True
            }
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Deep merge configurations
        self._deep_merge(self.config, user_config)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value