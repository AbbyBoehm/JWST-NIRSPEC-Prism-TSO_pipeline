__version__ = "0.1.0"
import os  # noqa
os.environ['CRDS_PATH'] = './crds_cache/jwst_ops'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from ..Juniper import Stage1, Stage2, Stage3, Stage4, Stage5