# Computational Process Python Tool

Python Tool for Modeling the Computational Process (Airbus Task 2). Includes the computational
process ontology, a MagicDraw to OpenMDAO interface based on OpenMBEE, and a GUI for the
visualization and comparison of computational process alternatives.

## Getting Started

### Installation

#### Creating a Virtual Environment [Optional]

It is recommended to create a dedicated virtual environment for this project. If using conda, an
environment configuration file `environment.yml` is provided for convenience. The conda
environment can be created by using:

    conda env create -f environment.yml

This should create a conda environment named `cppt` with all the dependencies of this project
, including the GUI.

#### Package Installation

Starting from the project main repository, the following installation options are available:

1.  **Basic Installation**: If you are _not_ planning on making any changes to the packages, i.e
    ., you only intend to run scripts and launch the GUI, then simply install using:

        pip install .
    
    _Note_: do not forget the (`.`) at the end.
    
    

2.  **Developer Installation**: If you _are_ planning on making changes to the package, you should
    install it in developer mode using `-e` flag:

        pip install -e .
        
    Developer mode allows the package to be install _in-place_, i.e., the source code is not
    moved to `site-packages`. As a result, changes made to the source code directly affect the
    package execution.

3.  **With GUI Dependencies** [optional]: To enforce the installation has the requisite package for
    the GUI, add the `[all]` option to the above installation command, such that:
     
        pip install .[all]

    _Note_: Installation with the `[all]` option simply check if the visualization tools are
    available, and install them if they are not. This will have no effect if the visualization tools are already installed.

#### Launching the Graphical User Interface (GUI)

This project GUI is built using
[Jupyter interactive widgets](https://github.com/jupyter-widgets/ipywidgets)
and is rendered as a web-app using [Voilà](https://github.com/voila-dashboards/voila).
The GUI can be launched with the following step:

1.  From the project main repository, launch the Voilà server by entering the `voila` command
    . _Note_: If a virtual environment is used, it should be first activated.

2.  Once the Voilà server is running, a web page should automatically open with a view of the
    current project folders. Navigate to the `examples` directory.

3.  In the browser window, select the `surrogate_scoping > surrogate_scoping.ipynb` or the
    `staged_beam_problem > c_dashboard.ipynb` files.

4.  The GUI should appear after a few seconds.

## Documentation

The documentation is a work-in-progress and is not available at this time. The reader is invited
to read the inline comments and the function/class/method docstrings.
