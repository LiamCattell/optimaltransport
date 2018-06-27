
Installation
============

Since the ``optrans`` package cannot yet be installed via pip, easy_install or Anaconda, we must add its directory to the Python path:

.. code-block:: python
	
	import sys
	sys.path.append('path/to/optimaltransport')

Once we have added the path to the package, we can import classes and functions as usual. For example:

.. code-block:: python

	from optrans.utils import signal_to_pdf
	from optrans.continuous import CDT
