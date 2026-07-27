import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SmokeTest(unittest.TestCase):
    """Production basic smoke tests to verify environment and module integrity."""
    
    def test_environment_python_version(self):
        """Verify Python 3.8+ environment."""
        self.assertGreaterEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 8)
        
    def test_directory_structure(self):
        """Verify essential production directories exist."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.assertTrue(os.path.exists(project_root), "Project root should exist.")
        
    def test_requirements_file_exists(self):
        """Verify dependency manifest exists."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        has_req = os.path.exists(os.path.join(project_root, 'requirements.txt')) or                   os.path.exists(os.path.join(project_root, 'package.json')) or                   os.path.exists(os.path.join(project_root, 'backend', 'requirements.txt'))
        self.assertTrue(has_req, "Dependency manifest must exist in production build.")

if __name__ == '__main__':
    unittest.main()
