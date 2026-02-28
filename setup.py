from setuptools import setup, find_packages

setup(
    name="swinssm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "monai-weekly[nibabel, tqdm, einops]",
        "matplotlib>=3.3.0",
        "einops>=0.3.0",
        "albumentations>=1.0.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.50.0",
        "captum>=0.6.0",
        "wandb",
        "scikit-learn",
        "pytorch-lightning",
        "scipy",
        "bitsandbytes"
    ],
    python_requires=">=3.7",
    author="Imroz R",
    author_email="imrozeshan@gmail.com",
    description="SwinSSM: Hybrid Swin Transformer and Mamba for 3D brain tumor segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yok4ai/SwinSSM.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 