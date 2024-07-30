from setuptools import setup

setup(name='juniper-package',
      version='1.0.4',
      description='JWST exoplanet time-series pipeline',
      long_description="Pipeline for handling James Webb Space Telescope time-series observations of exoplanet transits and eclipses.",
      author='Abby Boehm',
      packages=['juniper-package',
                'juniper.util',
                'juniper.stage1',
                'juniper.stage2',
                'juniper.stage3'],
      python_requires='>=3.8.0',
      install_requires=['scipy>=1.8.0', 'numpy', 'jwst>1.10.0', 'tqdm',
                        'matplotlib',],
      zip_safe=False)
