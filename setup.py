from setuptools import setup, find_packages

setup(
    name='vs10',
    version='0.0.1',
    url='https://github.com/slavzilla/vs10',
    description='Deep learning approach to VS10 benchmark',

    author='Slavko Kovacevic',
    author_email='skovacevic@ucg.ac.me',
    zip_safe=False,

    packages=find_packages(),
    install_requires=['tensorflow==2.8.0', 'librosa==0.7.2', 'numpy==1.19.5']
)