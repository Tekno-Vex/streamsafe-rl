"""
Export trained PyTorch PPO policy to ONNX format
ONNX enables low-latency inference in production (<3ms)
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.ppo_train import PolicyNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    state_dim: int = 10,
    action_dim: int = 5,
    opset_version: int = 14
):
    """
    Export PyTorch PPO policy to ONNX format
    """
    logger.info(f"Loading PyTorch model from {pytorch_model_path}")

    # Initialize model
    model = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)

    # Load checkpoint
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input for tracing
    dummy_state = torch.randn(1, state_dim)

    # Export to ONNX
    logger.info(f"Exporting model to ONNX (opset {opset_version})...")

    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_state,
        onnx_output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_probs', 'state_value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_probs': {0: 'batch_size'},
            'state_value': {0: 'batch_size'}
        }
    )

    logger.info(f"Model successfully exported to {onnx_output_path}")

    # Verify the ONNX model
    verify_onnx_model(onnx_output_path, dummy_state.numpy())

    return onnx_output_path

def verify_onnx_model(onnx_model_path: str, dummy_input: np.ndarray):
    """
    Verify ONNX model is valid and computable
    """
    logger.info("Verifying ONNX model...")

    # Check model validity
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid")

    # Test inference with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    # Run inference
    outputs = ort_session.run(output_names, {input_name: dummy_input})
    action_probs = outputs[0]
    state_value = outputs[1]

    logger.info(f"ONNX inference successful")
    logger.info(f"Action probabilities shape: {action_probs.shape}")
    logger.info(f"State value shape: {state_value.shape}")

    # Measure inference latency
    import time
    num_runs = 1000
    start = time.time()
    for _ in range(num_runs):
        ort_session.run(output_names, {input_name: dummy_input})
    elapsed = (time.time() - start) * 1000
    avg_latency = elapsed / num_runs

    logger.info(f"Average ONNX inference latency: {avg_latency:.3f}ms ({num_runs} runs)")

    if avg_latency < 3.0:
        logger.info("Latency requirement met (<3ms)")
    else:
        logger.warning(f"Latency {avg_latency:.3f}ms exceeds 3ms target")

def main():
    """
    Main export script
    """
    import argparse

    parser = argparse.ArgumentParser(description="Export PPO policy to ONNX format")
    parser.add_argument("--input", type=str, default="models/ppo_policy.pt", help="PyTorch model path")
    parser.add_argument("--output", type=str, default="models/ppo_policy.onnx", help="ONNX output path")
    args = parser.parse_args()

    # Export
    onnx_path = export_to_onnx(
        pytorch_model_path=args.input,
        onnx_output_path=args.output
    )

    logger.info("ONNX export complete")
    logger.info(f"Model ready for deployment: {onnx_path}")

if __name__ == "__main__":    
    main()