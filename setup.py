#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='OpenAI Research Project',
    author='Tyna Eloundou',
    author_email='mfe25@cornell.edu',
    url='https://github.com/feloundou/research-project',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
