import sys
import warnings
from pathlib import Path

# Add project root (parent of tests/) to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress known warnings from external dependencies
warnings.filterwarnings("ignore", category=UserWarning, message=".*model_name.*protected namespace.*")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, message=".*python_multipart.*")
