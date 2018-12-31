from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = ['matplotlib',
                     'graphviz',
                     'tensorflow',
                     'scikit-learn',
                     'tifffile',
                     'pillow',
                     'scipy',
                     'numpy',
                     'opencv-python>=3.3.0',
                     'torch',
                     'torchvision']

setup(
    name="rad_net",
    version="0.1.0",
    author="Mukesh Mithrakumar",
    author_email="mukesh.mithrakumar@jacks.sdstate.edu",
    description="PyTorch implementation of U-Net for biomedical image segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    url="https://github.com/mukeshmithrakumar/",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=REQUIRED_PACKAGES,
    classifiers=(
        "Development Status :: 0.1.0.dev1",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Version Control :: Git"

    ),
    entry_points={
        'console_scripts': [
            'unet-train = pytorch_unet.trainer.train:main',
            'unet-evaluate = pytorch_unet.trainer.evaluate:main',
            'unet-interpret = pytorch_unet.trainer.interpret:main',
        ]
    },
    python_requires='>=3',
)
