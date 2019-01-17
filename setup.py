from setuptools import find_packages, setup
from setuptools.extension import Extension


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
                     'torchvision',
                     'cython',
                     'psutil'
                     ]

extensions = [
    Extension(
        'pytorch_unet.optimize.c_extensions',
        ['pytorch_unet/optimize/c_extensions.pyx']
    ),
]

setup(
    name="radnet",
    version="0.1.0",
    author="Mukesh Mithrakumar",
    author_email="mukesh@mukeshmithrakumar.com",
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
        'Intended Audience :: Science/Research',
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
    entry_points={
        'console_scripts': [
            'radnet-train = pytorch_unet.trainer.train:main',
            'radnet-evaluate = pytorch_unet.trainer.evaluate:main',
            'radnet-interpret = pytorch_unet.trainer.interpret:main',
        ]
    },
    python_requires='>=3',

    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)
