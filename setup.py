from setuptools import setup, find_packages

setup(
    name='dnd',
    version='0.1.0',
    author='Michael Davies',
    author_email='michaelstoby@gmail.com',
    description='A simple tool for profiling and tracing PyTorch programs on NVIDIA GPUs',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'torch'],
    entry_points = {
        'console_scripts': [
            'dnd=dnd.run:main',
            'dnd-stat=dnd.run:stat',
        ],
    },
)
