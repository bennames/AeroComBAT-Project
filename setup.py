from setuptools import setup
import AeroComBAT

setup(
    name='AeroComBAT',
    description=AeroComBAT.__doc__,
    author='Ben Names',
    author_email='bennames@vt.edu',
    packages=['AeroComBAT','validations'],
    version='0.2.1',
    license='MIT',
    install_requires=[
        'numpy',
        'mayavi']
    )