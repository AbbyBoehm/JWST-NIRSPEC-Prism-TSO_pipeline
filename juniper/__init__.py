__version__ = "1.0.4"
import os  # noqa
os.environ['CRDS_PATH'] = './crds_cache/jwst_ops'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
from juniper import Stage1, Stage2, Stage3, Stage4, Stage5