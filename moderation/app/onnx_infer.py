"""
ONNX Runtime Integration for Production RL Inference
Provides low-latency (<3ms) inference with shadow mode support
"""

import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, Optional
from enum import IntEnum
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Discrete action space matching moderation actions"""
    IGNORE = 0
    WARN = 1
    TIMEOUT_60S = 2
    TIMEOUT_600S = 3
    BAN = 4


class ONNXInferenceEngine:
    """
    Production ONNX Runtime inference for PPO policy
    Features:
    - <3ms inference latency
    - Shadow mode support (no side effects)
    - Fallback to deterministic baseline
    """
    
    def __init__(
        self,
        model_path: str,
        shadow_mode: bool = True,
        enable_optimization: bool = True
    ):
        """
        Initialize ONNX Runtime inference engine
        
        Args:
            model_path: Path to ONNX model file
            shadow_mode: If True, log decisions without executing
            enable_optimization: Enable ONNX Runtime optimizations
        """
        self.model_path = model_path
        self.shadow_mode = shadow_mode
        
        logger.info(f"Loading ONNX model from {model_path}")
        logger.info(f"Shadow mode: {shadow_mode}")
        
        # Configure session options for performance
        sess_options = ort.SessionOptions()
        if enable_optimization:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Performance tracking
        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.max_latency_ms = 0.0
        
        logger.info(f"✓ ONNX Runtime initialized")
    
    def preprocess_state(self, state_dict: Dict) -> np.ndarray:
        """
        Convert state dictionary to model input tensor
        
        Expected state features (10-dim):
        - risk_score
        - message_count_24h
        - warning_count
        - timeout_count
        - account_age_days
        - follower_count
        - subscriber
        - moderator
        - channel_velocity
        - trust_score
        """
        features = np.array([
            state_dict.get('risk_score', 0.0),
            state_dict.get('message_count_24h', 0),
            state_dict.get('warning_count', 0),
            state_dict.get('timeout_count', 0),
            state_dict.get('account_age_days', 0),
            state_dict.get('follower_count', 0),
            1.0 if state_dict.get('subscriber', False) else 0.0,
            1.0 if state_dict.get('moderator', False) else 0.0,
            state_dict.get('channel_velocity', 0.0),
            state_dict.get('trust_score', 0.5)
        ], dtype=np.float32).reshape(1, -1)
        
        return features
    
    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray, float, float]:
        """
        Run ONNX inference
        
        Returns:
            action: Predicted action (int)
            action_probs: Action probability distribution
            state_value: Estimated state value
            latency_ms: Inference latency in milliseconds
        """
        start_time = time.perf_counter()
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: state}
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        action_probs = outputs[0][0]
        state_value = outputs[1][0][0]
        
        # Select action (greedy)
        action = int(np.argmax(action_probs))
        
        # Update metrics
        self.inference_count += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        return action, action_probs, float(state_value), latency_ms
    
    def infer(self, state_dict: Dict, deterministic_action: Optional[int] = None) -> Dict:
        """
        High-level inference API with shadow mode support
        
        Args:
            state_dict: State features dictionary
            deterministic_action: Baseline action from deterministic policy
        
        Returns:
            result: Dictionary containing RL action, probs, metrics
        """
        # Preprocess state
        state = self.preprocess_state(state_dict)
        
        # Run inference
        rl_action, action_probs, state_value, latency_ms = self.predict(state)
        
        # Convert to ActionType
        rl_action_type = ActionType(rl_action)
        
        result = {
            'rl_action': rl_action_type.name,
            'rl_action_idx': rl_action,
            'action_probs': action_probs.tolist(),
            'state_value': state_value,
            'latency_ms': latency_ms,
            'shadow_mode': self.shadow_mode,
            'deterministic_action': ActionType(deterministic_action).name if deterministic_action is not None else None,
            'agreement': rl_action == deterministic_action if deterministic_action is not None else None
        }
        
        if self.shadow_mode:
            logger.debug(
                f"Shadow inference | RL: {rl_action_type.name} | "
                f"Baseline: {result['deterministic_action']} | "
                f"Latency: {latency_ms:.2f}ms"
            )
        
        return result
    
    def get_metrics(self) -> Dict:
        """Get inference performance metrics"""
        if self.inference_count == 0:
            return {
                'inference_count': 0,
                'avg_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'latency_sla_met': True
            }
        
        avg_latency = self.total_latency_ms / self.inference_count
        
        return {
            'inference_count': self.inference_count,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': self.max_latency_ms,
            'latency_sla_met': avg_latency < 3.0
        }
    
    @classmethod
    def load_from_path(cls, model_path: str, shadow_mode: bool = True) -> 'ONNXInferenceEngine':
        """Factory method to load ONNX model"""
        if not Path(model_path).exists():
            logger.warning(f"ONNX model not found at {model_path}")
            return None
        
        return cls(model_path=model_path, shadow_mode=shadow_mode)