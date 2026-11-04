Installation
============

Follow these steps to set up the project on your local machine.

Prerequisites
-------------

catella is a Python package with a C++ extension that is compiled during
installation, so you need:

* Python 3.13 or newer
* A C++ compiler:

  * macOS: the Xcode Command Line Tools (``xcode-select --install``)
  * Debian/Ubuntu: ``sudo apt install build-essential`` (and the Python
    headers, e.g., ``python3.13-dev``, if you use the system Python)
  * Windows: Visual Studio Build Tools with the "Desktop development with
    C++" workload

Alternatively, Conda can provide both (see `Option B: Conda`_).

Step 1: Clone the Repository
----------------------------

First, clone the project to your local machine:

.. code-block:: bash

   git clone https://github.com/mchchiang/catella.git
   cd catella

Step 2: Install
---------------

Option A: pip
^^^^^^^^^^^^^

Create and activate a virtual environment, then install with pip:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install .

On Windows, activate the environment with ``.venv\Scripts\activate``
instead. pip builds the ``catella_cpp`` extension with scikit-build-core
and CMake, downloading CMake automatically if it is not already
installed. Use ``pip install -e .`` for an editable (development) install.

Option B: Conda
^^^^^^^^^^^^^^^

Use the provided ``environment.yml`` file to create a dedicated Conda
environment, which includes a C++ compiler. The default name of the
environment is ``catella``, but you can change this within the file.

.. code-block:: bash

   conda env create -f environment.yml
   conda activate catella

Step 3: Verify Installation
---------------------------

Check that the program is working correctly by running the help command:

.. code-block:: bash

   catella --help
