"""
Package trained model for deployment/sharing.
Creates a ZIP file with all necessary files.
"""

import shutil
from pathlib import Path
import zipfile

def package_model():
    """
    Create a deployment package with trained model.
    """
    print("Packaging trained model for deployment...")
    
    # Check if model exists
    if not Path('tokenizer_imdb.json').exists():
        print("❌ Error: tokenizer_imdb.json not found! Train the model first.")
        return
    
    weights_dir = Path('weights')
    if not weights_dir.exists() or len(list(weights_dir.glob('*.pt'))) == 0:
        print("❌ Error: No trained weights found! Train the model first.")
        return
    
    # Get latest weights
    weight_files = list(weights_dir.glob('*.pt'))
    latest_weight = max(weight_files, key=lambda p: p.stat().st_mtime)
    
    print(f"✓ Found trained model: {latest_weight.name}")
    print(f"✓ Found tokenizer: tokenizer_imdb.json")
    
    # Create deployment folder
    deploy_dir = Path('deployment_package')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy essential files
    essential_files = [
        'model.py',
        'config.py', 
        'dataset.py',
        'test.py',
        'tokenizer_imdb.json',
        'README.md'
    ]
    
    for file in essential_files:
        if Path(file).exists():
            shutil.copy(file, deploy_dir / file)
            print(f"✓ Copied {file}")
    
    # Copy trained weights
    weights_deploy = deploy_dir / 'weights'
    weights_deploy.mkdir()
    shutil.copy(latest_weight, weights_deploy / latest_weight.name)
    print(f"✓ Copied {latest_weight.name}")
    
    # Create minimal requirements.txt
    minimal_requirements = """torch>=2.0.0
datasets>=2.14.0
tokenizers>=0.13.0
tqdm>=4.65.0
"""
    with open(deploy_dir / 'requirements.txt', 'w') as f:
        f.write(minimal_requirements)
    print(f"✓ Created minimal requirements.txt")
    
    # Create deployment README
    deployment_readme = f"""# IMDB Sentiment Classifier - Trained Model

This package contains a trained transformer model for binary sentiment classification of movie reviews.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch datasets tokenizers tqdm
```

### 2. Test the Model

```python
python test.py
```

Choose option 1 for interactive mode or option 2 for examples.

### 3. Use in Your Code

```python
from test import load_trained_model, classify_review
from config import get_config

# Load model
config = get_config()
model, tokenizer, device = load_trained_model(config)

# Classify a review
review = "This movie was fantastic!"
prediction, confidence, probs = classify_review(
    review, model, tokenizer, config, device
)

print(f"Prediction: {{prediction}}")
print(f"Confidence: {{confidence:.2%}}")
```

## Model Details

- **Architecture**: Encoder-only Transformer (BERT-style)
- **Parameters**: ~12M
- **Training Data**: 22,500 IMDB reviews
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%

## Files Included

- `model.py` - Model architecture
- `config.py` - Configuration
- `dataset.py` - Dataset utilities
- `test.py` - Inference script
- `tokenizer_imdb.json` - Trained tokenizer
- `weights/{latest_weight.name}` - Trained model weights

## No Training Required!

The model is already trained. You can use it immediately for inference.

To retrain or fine-tune, you would need the full training script and IMDB dataset.
"""
    
    with open(deploy_dir / 'DEPLOYMENT_README.md', 'w') as f:
        f.write(deployment_readme)
    print(f"✓ Created DEPLOYMENT_README.md")
    
    # Create ZIP file
    zip_name = 'imdb_classifier_trained.zip'
    print(f"\nCreating ZIP file: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in deploy_dir.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(deploy_dir)
                zipf.write(file, arcname)
                print(f"  Added: {arcname}")
    
    # Cleanup
    shutil.rmtree(deploy_dir)
    
    # Get file size
    zip_size = Path(zip_name).stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Package created successfully!")
    print(f"📦 File: {zip_name}")
    print(f"📊 Size: {zip_size:.2f} MB")
    print(f"\n🚀 Share this file - no retraining needed!")
    print(f"   The recipient just needs to:")
    print(f"   1. Unzip the file")
    print(f"   2. Install dependencies (pip install -r requirements.txt)")
    print(f"   3. Run: python test.py")


if __name__ == "__main__":
    package_model()
