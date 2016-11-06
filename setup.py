from distutils.core import setup

import pypandoc

long_description = pypandoc.convert('README.md', 'rst')

setup(
    name='optimal',
    version='0.1.0',
    packages=['optimal'],

    author='Justin Lovinger',
    license='MIT',
    description="A python metaheuristic optimization library. Currently supports Genetic Algorithms, Gravitational Search, and Cross Entropy.",
    #long_description=long_description,
    keywords=['optimization', 'metaheuristic', 'genetic algorithm', 'GA',
              'gravitational search algorithm', 'GSA', 'cross entropy'],

    url='https://github.com/JustinLovinger/Optimal',
)