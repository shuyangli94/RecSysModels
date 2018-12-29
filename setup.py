from setuptools import setup, find_packages

# Get the long description from README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name = 'recsys_models',
    packages = find_packages(),
    version = '0.1.3',
    description = 'TensorFlow Recommender Systems Models for Implicit Feedback',
    author = 'Shuyang Li',
    author_email = 'shuyangli94@gmail.com',
    url = 'https://github.com/shuyangli94/RecSysModels',
    license = 'GPLv3+',
    keywords = ['recommender systems', 'recommender', 'recommendation system', 'tensorflow'],
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires = install_requires,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
)
