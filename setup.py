import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sciab',
    version='0.1.0',
    description='Benchmark for comparing (sample efficient) stable-controller inference algorithms',
    url='https://github.com/watakandai/sample-efficient-controller-benchmark',
    author='Kandai Watanabe',
    author_email='kandai.wata@gmail.com',
    license=license,
    packages=find_packages(exclude=('test', 'doc')),
    install_requires=read_requirements(),
)
