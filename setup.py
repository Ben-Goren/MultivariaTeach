from setuptools import setup, find_packages

setup(
    name="MultivariaTeach",
    version="0.1.7",
    description="A collection of tools intended for students of multivariate analysis",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Ben Goren",
    author_email="bgoren@asu.edu",
    url="https://github.com/Ben-Goren/MultivariaTeach",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.5.0",
        "pandas>=1.2.0",
    ],
    python_requires=">=3.7",
)
