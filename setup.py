from setuptools import setup
import AeroComBAT

setup(
    name='AeroComBAT',
    description=AeroComBAT.__doc__,
    author='Ben Names',
    author_email='bennames@vt.edu',
    packages=['AeroComBAT'],
    version='1.0a1',
    license='MIT',
    url='https://github.com/bennames/AeroComBAT-Project',
    install_requires=[
        'numpy',
        'mayavi',
        'numba'],
    classifiers=[
    'Development Status :: 3 - Alpha',

    'Intended Audience :: Aerospace Stress Analysts',

     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2.7',],
    )