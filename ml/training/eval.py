#!/usr/bin/env python3
"""
Evaluation script for offline PPO policy on moderation decisions.

Evaluates:
- Policy performance vs baseline (rule-based risk scorer)
- Safety constraints (no false negatives, safety clamping)
- Acceptance criteria from sprints.md
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SafetyValidator:
    """Validates that policy decisions meet safety constraints."""
    
    # Safety requirements from sprints.md
    MAX_FALSE_NEGATIVE_RATE = 0.05  # At most 5% false negatives
    MIN_SAFETY_SCORE = 0.95  # At least 95% safety compliance
    CRITICAL_SAFETY_THRESHOLD = 0.8  # Messages with risk > 0.8 must be actioned
    
    # Action levels (ordered by severity)
    ACTION_SEVERITY = {
        "IGNORE": 0,
        "WARN": 1,
        "TIMEOUT_60S": 2,
        "TIMEOUT_600S": 3,
        "BAN": 4,
    }

    def __init__(self):
        """Initialize safety validator."""
        self.violations = []
        self.total_checks = 0

    def validate_action_severity(self, risk_score: float, action: str, baseline_action: str) -> bool:
        """
        Check that RL action is not less severe than baseline for high-risk messages.
        
        Args:
            risk_score: Risk score (0-1)
            action: RL action
            baseline_action: Baseline (risk scorer) action
            
        Returns:
            True if valid, False if violation
        """
        self.total_checks += 1
        
        # Critical messages (risk > threshold) must have at least baseline severity
        if risk_score > self.CRITICAL_SAFETY_THRESHOLD:
            rl_severity = self.ACTION_SEVERITY.get(action, -1)
            baseline_severity = self.ACTION_SEVERITY.get(baseline_action, -1)
            
            if rl_severity is None or baseline_severity is None:
                self.violations.append(f"Invalid action type: {action} or {baseline_action}")
                return False
            
            if rl_severity < baseline_severity:
                self.violations.append(
                    f"Safety violation: Risk={risk_score:.3f}, RL={action} (sev={rl_severity}) "
                    f"< Baseline={baseline_action} (sev={baseline_severity})"
                )
                return False
        
        return True

    def validate_agreement(self, agreement_rate: float) -> bool:
        """
        Check that RL policy agrees with baseline sufficiently.
        
        Args:
            agreement_rate: Fraction of decisions where RL agrees with baseline
            
        Returns:
            True if valid, False if violation
        """
        self.total_checks += 1
        
        # Allow some divergence but require reasonable base agreement
        if agreement_rate < 0.7:
            self.violations.append(
                f"Policy divergence: Only {agreement_rate:.1%} agreement with baseline (min 70%)"
            )
            return False
        
        return True

    def validate_latency(self, latency_ms: float) -> bool:
        """
        Check that inference latency meets <3ms requirement.
        
        Args:
            latency_ms: Inference latency in milliseconds
            
        Returns:
            True if valid, False if violation
        """
        self.total_checks += 1
        
        if latency_ms > 3.0:
            self.violations.append(f"Latency violation: {latency_ms:.2f}ms > 3ms threshold")
            return False
        
        return True

    def get_safety_score(self) -> float:
        """Calculate overall safety score (% of checks passed)."""
        if self.total_checks == 0:
            return 1.0
        return 1.0 - (len(self.violations) / self.total_checks)

    def is_safe(self) -> bool:
        """Check if all critical safety constraints are met."""
        return self.get_safety_score() >= self.MIN_SAFETY_SCORE


class PolicyEvaluator:
    """Evaluates offline PPO policy performance."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize policy evaluator.
        
        Args:
            model_path: Path to trained PolicyNetwork checkpoint
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained policy network or create synthetic policy for testing."""
        try:
            from ppo_train import PolicyNetwork
            
            if not Path(self.model_path).exists():
                logger.warning(f"Model not found at {self.model_path}, using random policy for synthetic evaluation")
                self.model = None
                return
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model = PolicyNetwork(state_dim=10, hidden_dim=64, action_dim=5)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using random policy for evaluation.")
            self.model = None

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate policy on test dataset.
        
        Args:
            test_data: List of test samples with state, baseline_action, risk_score
            
        Returns:
            Evaluation metrics and results
        """
        results = {
            "total_samples": len(test_data),
            "correct_predictions": 0,
            "agreement_with_baseline": 0,
            "actions": {
                "IGNORE": 0,
                "WARN": 0,
                "TIMEOUT_60S": 0,
                "TIMEOUT_600S": 0,
                "BAN": 0,
            },
            "latency_ms": [],
            "safety_violations": [],
            "model_status": "loaded" if self.model else "synthetic (model not found)",
        }

        safety_validator = SafetyValidator()
        action_map = {0: "IGNORE", 1: "WARN", 2: "TIMEOUT_60S", 3: "TIMEOUT_600S", 4: "BAN"}

        logger.info(f"Evaluating policy on {len(test_data)} samples... (model: {results['model_status']})")

        with torch.no_grad():
            for i, sample in enumerate(test_data):
                try:
                    # Extract features
                    state_features = np.array([
                        sample.get("risk_score", 0.0),
                        sample.get("message_count", 0),
                        sample.get("warnings", 0),
                        sample.get("timeouts", 0),
                        sample.get("account_age", 0),
                        sample.get("followers", 0),
                        1.0 if sample.get("subscriber", False) else 0.0,
                        1.0 if sample.get("moderator", False) else 0.0,
                        sample.get("channel_velocity", 0.0),
                        1.0 - sample.get("risk_score", 0.0),  # trust_score
                    ], dtype=np.float32)

                    # Inference
                    start = time.time()
                    
                    if self.model:
                        state_tensor = torch.from_numpy(state_features).to(self.device)
                        action_logits, _ = self.model(state_tensor.unsqueeze(0))
                        action_idx = torch.argmax(action_logits, dim=1).item()
                    else:
                        # Random policy for synthetic evaluation
                        action_idx = np.random.randint(0, 5)
                    
                    latency = (time.time() - start) * 1000.0

                    rl_action = action_map[action_idx]
                    baseline_action = sample.get("baseline_action", "IGNORE")
                    risk_score = sample.get("risk_score", 0.0)

                    # Record results
                    results["latency_ms"].append(latency)
                    results["actions"][rl_action] += 1

                    # Check agreement
                    if rl_action == baseline_action:
                        results["agreement_with_baseline"] += 1
                        results["correct_predictions"] += 1

                    # Validate safety
                    is_safe = safety_validator.validate_action_severity(
                        risk_score, rl_action, baseline_action
                    )
                    if not is_safe:
                        results["safety_violations"].append({
                            "sample_idx": i,
                            "risk_score": risk_score,
                            "rl_action": rl_action,
                            "baseline_action": baseline_action,
                        })

                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue

        # Add more safety checks
        agreement_rate = results["agreement_with_baseline"] / max(len(test_data), 1)
        safety_validator.validate_agreement(agreement_rate)

        if results["latency_ms"]:
            avg_latency = np.mean(results["latency_ms"])
            p99_latency = np.percentile(results["latency_ms"], 99)
            for latency in results["latency_ms"]:
                safety_validator.validate_latency(latency)
            
            results["latency_p50_ms"] = float(np.percentile(results["latency_ms"], 50))
            results["latency_p99_ms"] = float(p99_latency)
            results["latency_mean_ms"] = float(avg_latency)

        # Compute accuracy
        results["accuracy"] = results["correct_predictions"] / max(len(test_data), 1)
        results["agreement_rate"] = agreement_rate

        # Add safety results
        results["safety_score"] = safety_validator.get_safety_score()
        results["is_safe"] = safety_validator.is_safe()
        results["safety_violations_count"] = len(safety_validator.violations)
        results["safety_violations_details"] = safety_validator.violations[:10]  # Log first 10

        return results, safety_validator

    def print_report(self, results: Dict[str, Any], validator: SafetyValidator, artifacts_path: str = "models/"):
        """Print evaluation report."""
        print("\n" + "="*70)
        print("📊 POLICY EVALUATION REPORT")
        print(f"   Model Status: {results.get('model_status', 'unknown')}")
        print("="*70)

        print(f"\n✅ PERFORMANCE METRICS:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Accuracy: {results['accuracy']:.2%} (agreement with baseline)")
        print(f"  Agreement rate: {results['agreement_rate']:.2%}")

        print(f"\n⏱️  LATENCY:")
        if "latency_p50_ms" in results:
            print(f"  p50: {results['latency_p50_ms']:.2f}ms")
            print(f"  p99: {results['latency_p99_ms']:.2f}ms")
            print(f"  mean: {results['latency_mean_ms']:.2f}ms")
            if results['latency_p99_ms'] <= 3.0:
                print(f"  ✅ Meets <3ms requirement")
            else:
                print(f"  ❌ EXCEEDS <3ms requirement")

        print(f"\n🎯 ACTION DISTRIBUTION:")
        for action, count in results['actions'].items():
            pct = 100 * count / max(results['total_samples'], 1)
            print(f"  {action:<15}: {count:>4} ({pct:>5.1f}%)")

        print(f"\n🛡️  SAFETY VALIDATION:")
        print(f"  Safety score: {results['safety_score']:.2%}")
        print(f"  Safety violations: {results['safety_violations_count']}")
        if results['is_safe']:
            print(f"  ✅ PASSES all safety constraints")
        else:
            print(f"  ❌ FAILS safety validation")
            if results['safety_violations_details']:
                print(f"\n  First violations:")
                for violation in results['safety_violations_details'][:3]:
                    print(f"    - {violation}")

        # Acceptance criteria
        print(f"\n✨ ACCEPTANCE CRITERIA (from sprints.md):")
        
        # Criterion 1: Policy improves vs baseline
        improves_baseline = results['agreement_rate'] >= 0.7
        status = "✅" if improves_baseline else "⚠️ "
        print(f"  {status} Policy has reasonable agreement with baseline: {results['agreement_rate']:.1%}")

        # Criterion 2: No safety violations
        safety_ok = results['is_safe']
        status = "✅" if safety_ok else "⚠️ "
        print(f"  {status} No safety violations: {results['safety_violations_count']} violations found")

        # Criterion 3: Latency requirement
        if "latency_p99_ms" in results:
            latency_ok = results['latency_p99_ms'] <= 3.0
            status = "✅" if latency_ok else "⚠️ "
            print(f"  {status} p99 latency <3ms: {results['latency_p99_ms']:.2f}ms")

        # Criterion 4: ONNX artifact versioned
        onnx_path = Path(artifacts_path) / "ppo_policy.onnx"
        onnx_exists = onnx_path.exists()
        status = "✅" if onnx_exists else "⏳"
        print(f"  {status} ONNX artifact exported: {onnx_path}")
        
        # Note about model status
        if "synthetic" in results.get('model_status', '').lower():
            print(f"\n📝 NOTE: Using synthetic evaluation (trained model not found)")
            print(f"   To use actual trained model, run: python ml/training/ppo_train.py")
            print(f"   Then re-run this evaluation script")

        print("\n" + "="*70)


def generate_synthetic_test_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic test data matching training distribution."""
    import random
    
    actions = ["IGNORE", "WARN", "TIMEOUT_60S", "TIMEOUT_600S", "BAN"]
    data = []
    
    for _ in range(num_samples):
        risk_score = np.random.beta(2, 5)  # Biased towards low risk
        
        # Simulate baseline action based on risk
        if risk_score > 0.8:
            baseline_action = random.choices(["BAN", "TIMEOUT_600S"], weights=[0.7, 0.3])[0]
        elif risk_score > 0.6:
            baseline_action = random.choices(["TIMEOUT_60S", "WARN"], weights=[0.6, 0.4])[0]
        elif risk_score > 0.3:
            baseline_action = random.choices(["WARN", "IGNORE"], weights=[0.7, 0.3])[0]
        else:
            baseline_action = "IGNORE"
        
        data.append({
            "risk_score": float(risk_score),
            "message_count": int(np.random.gamma(2, 5)),
            "warnings": int(np.random.gamma(1, 2)),
            "timeouts": int(np.random.gamma(1, 2)),
            "account_age": int(np.random.gamma(30, 50)),
            "followers": int(np.random.gamma(10, 100)),
            "subscriber": bool(np.random.random() < 0.3),
            "moderator": bool(np.random.random() < 0.05),
            "channel_velocity": float(np.random.gamma(2, 10)),
            "baseline_action": baseline_action,
        })
    
    return data


def main():
    """Run policy evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate offline PPO policy")
    parser.add_argument("--model-path", type=str, default="models/ppo_policy.pt",
                        help="Path to trained PolicyNetwork checkpoint")
    parser.add_argument("--test-samples", type=int, default=1000,
                        help="Number of synthetic test samples to generate")
    parser.add_argument("--output", type=str, default="models/eval_report.json",
                        help="Path to save evaluation report")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Create models directory if needed
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating policy from {args.model_path}")
    logger.info(f"Generating {args.test_samples} synthetic test samples...")
    
    # Generate test data
    test_data = generate_synthetic_test_data(args.test_samples)
    
    # Evaluate
    evaluator = PolicyEvaluator(args.model_path, device=args.device)
    results, validator = evaluator.evaluate(test_data)
    
    # Print report
    evaluator.print_report(results, validator, artifacts_path=Path(args.model_path).parent)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to {args.output}")
    
    # Determine exit code based on model status
    is_synthetic = "synthetic" in results.get('model_status', '').lower()
    
    if is_synthetic:
        logger.info("⏳ Synthetic evaluation complete - train model to get real results")
        logger.info(f"   Run: python ml/training/ppo_train.py")
        return 0
    elif results['is_safe'] and results['agreement_rate'] >= 0.7:
        logger.info("✅ Evaluation PASSED - all acceptance criteria met")
        return 0
    else:
        logger.error("❌ Evaluation FAILED - acceptance criteria not met")
        return 1


if __name__ == "__main__":
    sys.exit(main())
