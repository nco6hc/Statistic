"""
Verify that all required packages are installed correctly
"""

import sys

print("=" * 70)
print("Verifying Package Installation".center(70))
print("=" * 70)

required_packages = {
    'pandas': 'Data processing',
    'numpy': 'Numerical computations',
    'openpyxl': 'Excel file reading',
    'torch': 'Deep learning (LSTM)',
    'sklearn': 'Machine learning utilities',
    'matplotlib': 'Visualization'
}

print(f"\nPython Version: {sys.version.split()[0]}")
print(f"Required: 3.10 or higher")

if sys.version_info < (3, 10):
    print("WARNING: Python version is below 3.10!")
else:
    print("✓ Python version OK")

print("\n" + "-" * 70)
print("Checking Required Packages:")
print("-" * 70)

all_installed = True

for package, description in required_packages.items():
    try:
        if package == 'sklearn':
            module = __import__('sklearn')
        else:
            module = __import__(package)
        
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package:<15} {version:<15} ({description})")
    except ImportError:
        print(f"✗ {package:<15} NOT INSTALLED    ({description})")
        all_installed = False

print("-" * 70)

if all_installed:
    print("\n✓ All packages installed successfully!")
    print("\nYou can now run:")
    print("  - LSTM model: cd dl_pipeline && python main_pipeline.py")
    print("  - Monte Carlo: cd monte_carlo_pipeline && python monte_carlo_model.py")
else:
    print("\n✗ Some packages are missing!")
    print("\nTo install missing packages, run:")
    print("  pip install -r requirements.txt")

print("\n" + "=" * 70)
