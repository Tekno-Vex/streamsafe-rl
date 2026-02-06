import unittest
from app.safety_clamp import SafetyClamp, ActionType

class TestSafetyClamp(unittest.TestCase):
    def setUp(self):
        self.clamp = SafetyClamp()

    def test_moderator_immunity(self):
        """Test that moderators are immune to bans"""
        # Attempt to BAN a moderator
        result = self.clamp.clamp(ActionType.BAN, user_trust_score=0.5, is_moderator=True)
        # Should be downgraded to IGNORE
        self.assertEqual(result, ActionType.IGNORE)

    def test_high_trust_downgrade(self):
        """Test that high trust users get lighter punishments"""
        # Attempt to BAN a trusted user
        result = self.clamp.clamp(ActionType.BAN, user_trust_score=0.9, is_moderator=False)
        # Should be downgraded (Not BAN)
        self.assertNotEqual(result, ActionType.BAN)

if __name__ == '__main__':
    unittest.main()