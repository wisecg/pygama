Contribution guidelines
=======================

The following rules and conventions have been established for the package
development and are enforced throughout the entire code base. Merge requests
that do not comply to the following directives will be rejected.

Coding style
------------

*In progress...*

Testing
-------

* The *pygama* test suite is available below ``test/``
* Open source best practices are adopted. In particular:

  * Tests are provided for each class/method separately (*micro-testing*) e.g.
    (for each single DSP processor)
  * High-level tests of realistic applications are provided (e.g. full
    production of a real data sample)

* Unit tests are automatically run for every push event and pull request to
  the remote Git repository on a build server (currently GitHub actions)
* Every pull request must pass all tests before being approved for merging
* Pull request authors are required to provide tests with sufficient coverage
  for every proposed change or addition. If necessary, global tests (e.g. tests
  of full production chain) should be updated.

*In progress...*

Versioning
----------

* `Semantic versioning <https://semver.org>`_ is adopted. The version string
  uses the ``MAJOR.MINOR.PATCH`` format.
* The version string is manually specified in ``pygama/version.py``. If needed
  elsewhere in the source code (e.g. in ``setup.py``), must be read in from here.
* To release a new minor or major version, the following procedure should be
  followed:

  * The *pygama* version is updated in ``pygama/version.py``
  * A new branch with name ``releases/vMAJOR.MINOR`` (note the ``v``) containing
    the code at the intended stage is created
  * The commit is tagged with a descriptive message: ``git tag vMAJOR.MINOR.0 -m 'short descriptive message here'`` (note the ``v``)
  * Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

* To release a new patch version, the following procedure should be followed:

  * The *pygama* version is updated in ``pygama/version.py``
  * A commit with the patch is created on the relevant release branch
    ``releases/vMAJOR.MINOR``
  * The commit is tagged: ``git tag vMAJOR.MINOR.PATCH`` (note the ``v``)
  * Changes are pushed to the remote: ``git push --tags origin releases/vMAJOR.MINOR``

Documentation
-------------

* Documentation should be exclusively available on the package website
  https://legend-exp.github.io/pygama. No READMEs, GitHub/LEGEND wiki pages
  should be written.
* Jupyter notebooks should be added to the Git repository below ``tutorials/``.
* Pull request authors are required to provide sufficient documentation (mainly
  as *docstrings*) for every proposed change or addition.
