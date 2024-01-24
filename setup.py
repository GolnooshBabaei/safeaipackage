from setuptools import setup, find_packages
import codecs
import os

VERSION = '1.1.4'
DESCRIPTION = 'SAFE AI package to measure robuStness, Accuracy, Fairness, and Explainability of an AI model'

# Setting up
setup(
    name="safeaipackage",
    version=VERSION,
    author="Golnoosh Babaei",
    author_email="<golnoosh.babaei93@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
