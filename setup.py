
from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GCProc',
    version='0.1.0',
    packages=['src', 'src.gcproc', 'src.gcproc.test', 'src.gcproc.utils', 'src.gcproc.experimential'],
    # packages=setuptools.find_packages(where="src"),
    url='https://github.com/thisismygitrepo/gcprocpy',
    license='MIT',
    author=['Alex Al-Saffar', 'David Banh'],
    author_email='programmer@usa.com',
    description='Python Implementation of the R-based GCProc',
    long_description=long_description,
    python_requires=">=3.6",

)
