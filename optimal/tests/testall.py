# Perform a little hack to make sure the optimal library is visible from this script
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir) 

import pytest

pytest.main('-m "not webtest and not slowtest"')