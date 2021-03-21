from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='coinpy',
    description='CoinPy is an open-source project for making cryptocurrency available for everyone.',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'test.*', 'examples.*')),
    install_requires=['pandas',
                      'cbpro',
                      'matplotlib',
                      'pytest'
                      ],
    extras_require={"dev": ["pytest>= 3.7"]},
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/Demirrr/coinpy',
    classifiers=[
        "Programming Language :: Python :: 3.8.5",
        "License :: OSI Approved :: MIT License", ],
    python_requires='>=3.8.5',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
