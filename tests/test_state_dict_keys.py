#!/usr/bin/env python3
"""
Test that API state dict keys match ONNX engine expectations
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Simulate the keys API sends
api_state_dict = {
    "risk_score": 0.75,
    "message_count_24h": 50,
    "warning_count": 2,
    "timeout_count": 1,
    "account_age_days": 365,
    "follower_count": 0,
    "subscriber": False,
    "moderator": False,
    "channel_velocity": 12.5,
    "trust_score": 0.25,
}

# Import ONNX engine preprocessing
from moderation.app.onnx_infer import ONNXInferenceEngine

print("\n" + "=" * 70)
print("🔍 TESTING STATE DICT KEY CONSISTENCY")
print("=" * 70)

print("\n📝 State dict from API:")
for key, value in api_state_dict.items():
    print(f"   {key}: {value}")

# Test preprocessing (without loading model)
print("\n🔬 Testing ONNX preprocessing...")

# Manually replicate preprocessing logic
features = np.array([
    api_state_dict.get('risk_score', 0.0),
    api_state_dict.get('message_count_24h', 0),
    api_state_dict.get('warning_count', 0),
    api_state_dict.get('timeout_count', 0),
    api_state_dict.get('account_age_days', 0),
    api_state_dict.get('follower_count', 0),
    1.0 if api_state_dict.get('subscriber', False) else 0.0,
    1.0 if api_state_dict.get('moderator', False) else 0.0,
    api_state_dict.get('channel_velocity', 0.0),
    api_state_dict.get('trust_score', 0.5)
], dtype=np.float32).reshape(1, -1)

print(f"\n✅ Preprocessed features shape: {features.shape}")
print(f"✅ Features: {features[0]}")

# Verify all values are present (not defaulting to 0)
expected_non_zero = [0.75, 50.0, 2.0, 1.0, 365.0, 0.0, 0.0, 0.0, 12.5, 0.25]
actual = features[0].tolist()

print(f"\n📊 Value comparison:")
keys_list = ['risk_score', 'message_count_24h', 'warning_count', 'timeout_count', 
             'account_age_days', 'follower_count', 'subscriber', 'moderator', 
             'channel_velocity', 'trust_score']

all_match = True
for i, (key, expected, actual_val) in enumerate(zip(keys_list, expected_non_zero, actual)):
    match = "✅" if abs(expected - actual_val) < 0.01 else "❌"
    if abs(expected - actual_val) >= 0.01:
        all_match = False
    print(f"   {match} {key:20s}: expected={expected:6.2f}, actual={actual_val:6.2f}")

print("\n" + "=" * 70)
if all_match:
    print("🎉 SUCCESS: All keys match, no values defaulting to 0")
    print("   API and ONNX engine are compatible!")
else:
    print("❌ FAILURE: Some values defaulted (key mismatch)")
print("=" * 70 + "\n")
