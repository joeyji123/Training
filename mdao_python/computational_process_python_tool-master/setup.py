from setuptools import setup, find_packages

setup(
    name='ComputationalProcessPythonTool',
    version='0.0.2',
    author='Raphael Gautier <raphael.gautier@gatech.edu>, '
           'Christian Perron <cperron7@gatech.edu>',
    description='Python Tool for Modeling the Computational Process (Airbus Task 2).',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'requests',
        'numpy',
        'openmdao>=3.0',
        'ipywidgets',
        'traitlets',
        'traittypes',
        'plotly',
        'ipytree',
    ],
    extras_require={
        'all': [
            'jupyter',
            'voila',
        ]
    },
    python_requires='>=3.6, <3.8'
)
