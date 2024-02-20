from setuptools import setup, find_packages
from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSION = '2.2.5'
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
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
