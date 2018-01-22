from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='behavior2vec',
      version='0.1',
      description='The behavior2Vec model',
      long_description='This is a package to run the behavior2vec model',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='Word2Vec, Behavior2Vec',
      url='https://github.com/ncu-dart/behavior2vec',
      author='Hung-Hsuan Chen',
      author_email='hhchen@ncu.edu.tw',
      license='MIT',
      packages=['behavior2vec'],
      install_requires=[
            'numpy',
            'scipy',
            'gensim',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      scripts=[
            'bin/b2v-train.py',
            'bin/b2v-most-similar-behavior.py',
            'bin/b2v-most-similar-item.py',
      ],
)
