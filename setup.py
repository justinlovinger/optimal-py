from distutils.core import setup

# Convert README.md to long description
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")  # YOU NEED THIS LINE
except (ImportError, OSError, IOError):
    print("Pandoc not found. Long_description conversion failure.")
    import io
    # pandoc is not installed, fallback to using raw contents
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='optimal',
    version='0.2.0',
    packages=['optimal', 'optimal.algorithms'],
    # Include example and test files
    package_data={'optimal': ['examples/*.py', 'tests/*.py', 'tests/algorithms/*.py']},
    # Include readme
    data_files=[('', ['README.md'])],
    # Dependencies
    install_requires=[
        'numpy'
    ],

    # Metadata
    author='Justin Lovinger',
    license='MIT',
    description="A python metaheuristic optimization library. Currently supports Genetic Algorithms, Gravitational Search, and Cross Entropy.",
    long_description=long_description,
    keywords=['optimization', 'metaheuristic', 'genetic algorithm', 'GA',
              'gravitational search algorithm', 'GSA', 'cross entropy'],

    url='https://github.com/JustinLovinger/Optimal',
)