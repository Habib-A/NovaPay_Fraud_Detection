# Fixing NumPy Import Error

If you're encountering the `numpy.core.multiarray failed to import` error, follow these steps:

## Option 1: Reinstall NumPy (Recommended)

In your conda environment, run:

```bash
conda install numpy=1.24.3 -y
conda install scikit-learn -y
pip install --upgrade --force-reinstall shap
```

## Option 2: Complete Environment Fix

If Option 1 doesn't work, try a more comprehensive fix:

```bash
# Uninstall problematic packages
pip uninstall numpy scikit-learn shap pyarrow -y
conda uninstall numpy scikit-learn pyarrow -y

# Reinstall with conda (better for binary compatibility)
conda install numpy=1.24.3 scikit-learn=1.3.0 -y
conda install pyarrow -y

# Install SHAP with pip
pip install shap==0.42.1
```

## Option 3: Create Fresh Environment (If all else fails)

```bash
# Create new conda environment
conda create -n novapay python=3.10 -y
conda activate novapay

# Install packages
pip install -r requirements.txt
```

## Verify Installation

After fixing, verify with:

```python
python -c "import numpy; import sklearn; import shap; print('All imports successful!')"
```

Then run the app:
```bash
streamlit run app.py
```

