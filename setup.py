from setuptools import setup, find_packages

setup(
    name='samplifi',
    version='v1.0.0',
    packages=find_packages(),
    install_requires=[
        'basic_pitch==0.3.0',
        'gin==0.1.6',
        'librosa==0.10.1',
        'matplotlib==3.8.3',
        'mir-eval==0.7',
        'mirdata==0.3.8',
        'numpy==1.23.5',
        'pandas==2.2.1',
        'pretty_midi==0.2.10',
        'pyclarity==0.4.1',
        'scipy==1.12.0',
        'setuptools==68.2.2',
        'tensorflow==2.16.1'
    ],
    author='hyve9',
    author_email='afb8252@nyu.edu',
    description='Add harmonic content to music to help hearing impaired listeners hear music better.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hyve9/samplifi',
)
