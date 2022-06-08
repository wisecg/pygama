Installation
============

Install on local systems with:

.. code-block:: console

    $ git clone https://github.com/legend-exp/pygama
    $ pip install pygama

Append the ``-e`` flag (editable mode) if you plan to develop *pygama*. If you
do not have admin rights you can install pygama as a normal user:

.. code-block:: console

    $ pip install pygama --user

and make sure that your ``PYTHONPATH`` environment variable is aware of the
install location. Optionally-required dependencies can be also collected:

.. code-block:: console

    $ pip install pygama[daq]  # if you need to convert DAQ files to LH5
    $ pip install pygama[test] # if you want to run package test files
    $ pip install pygama[docs] # if you need to build documentation

To uninstall pygama run:

.. code-block:: console

    $ pip uninstall pygama
