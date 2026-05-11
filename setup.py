from setuptools import setup, find_packages

setup(
    name="kanade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'anndata>=0.9.2',
        # 'torch>=1.11.0',
        # 'torchvision>=0.12.0',
        # 'torchaudio>=0.11.0',
        'scikit-learn>=1.3.2',
        'numpy>=1.24.4',
        'tqdm>=4.66.1'
    ]
)
