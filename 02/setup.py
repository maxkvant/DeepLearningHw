from distutils.core import setup

setup(
    name='ResNeXt',
    version='0.1dev',
    packages=['resnext',],
    license='MIT license',
    long_description=open('README.txt').read(),
    requires=['torch', 'tensorboardX', 'pytest', 'numpy']
)