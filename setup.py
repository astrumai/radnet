from distutils.command.build_ext import build_ext as distutils_build_ext

from setuptools import find_packages, setup, Command


class BuildRun(Command):
    description = distutils_build_ext.description
    user_options = distutils_build_ext.user_options
    boolean_options = distutils_build_ext.boolean_options
    help_options = distutils_build_ext.help_options

    def __init__(self, dist, *args, **kwargs):
        super().__init__(dist, **kwargs)
        from setuptools.command.build_ext import build_ext as setuptools_build_ext

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = setuptools_build_ext(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


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
                     'pytest']

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
    cmdclass={'build_run': BuildRun}
)
