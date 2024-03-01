from setuptools import setup, find_packages
from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent

# Open README.md with UTF-8 encoding
with open(this_directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()
    
VERSION = '0.4.0'
DESCRIPTION = 'SAFE AI package to measure risks of AI models'

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
