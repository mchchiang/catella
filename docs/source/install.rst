Installation
============

Follow these steps to set up the project on your local machine.

Prerequisites
-------------

Before you begin, ensure you have `Conda <https://docs.anaconda.com>`
installed on your system.

Step 1: Clone the Repository
----------------------------

First, clone the project to your local machine:

.. code-block:: bash

   git clone https://git.ecdf.ed.ac.uk/cchiang2/nucmc.git

Step 2: Create the Environment
------------------------------

Use the provided `environment.yml` file to create a dedicated Conda
environment. This ensures all dependencies are correctly managed. The default
name of the environment is `catella`, but you can change this within the
file.
file.

.. code-block:: bash

   conda env create -f environment.yml

Step 3: Activate the Environment
--------------------------------

Once the environment is created, activate it using:

.. code-block:: bash

   conda activate catella

Step 4: Verify Installation
---------------------------

Check that the program is working correctly by running the help command:

.. code-block:: bash

   catella --help

