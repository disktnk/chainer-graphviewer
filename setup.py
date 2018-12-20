from setuptools import setup


setup(
    name='chainer-graphviewer',
    version='0.1.0',
    description='graph viewer on Jupyter Notebook',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Daisuke Tanaka',
    author_email='duaipp@gmail.com',
    url='https://github.com/disktnk/chainer-graphviewer',
    packages=['graphviewer'],
    install_requires=[
        'numpy>=1.15.4',
        'protobuf>=3.6.1',
    ],
    test_require=[],
)
