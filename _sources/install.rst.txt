Installing pygama
=================

Install on local systems with: ::

    > git clone https://github.com/legend-exp/pygama
    > pip install -e pygama

You can omit the ``-e`` flag (editable mode) if you do not plan to develop
pygama. If you do not have admin rights (e.g. at NERSC) you can install pygama
as a normal user: ::

    > pip install -e pygama --user

To uninstall pygama run: ::

    > pip uninstall pygama

To run pygama at NERSC (and set up JupyterHub), we have additional instructions
`at this link <https://github.com/legend-exp/legend/wiki/Computing-Resources-at-NERSC#configuring-jupyter--nersc>`_.
