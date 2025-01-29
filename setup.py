from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "seaborn"
    ],
    description="A package for training, testing, and analyzing spectrograms.",
    author="Your Name",
    author_email="your_email@example.com",
)
