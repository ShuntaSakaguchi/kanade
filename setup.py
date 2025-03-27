from setuptools import setup, find_packages

setup(
    name="kanade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scanpy>=1.9.6',
        'torch>=1.11.0+cu113',
        'torchvision>=0.12.0',
        'torchaudio>=0.11.0',
    ],
    extras_require={
        "gpu": [
            "torch>=1.11.0+cu113",
            'torchvision>=0.12.0+cu113',
            'torchaudio>=0.11.0+cu113'
        ],
    }
)
