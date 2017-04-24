from setuptools import setup

setup(name='pyAccuRT',
      version='1.0',
      description='Reading and plotting results from AccuRT',
      author='Torbjoern Taskjelle',
      author_email='totaskj@gmail.com',
      url='https://github.com/TorbjornT/pyAccuRT',
      license='MIT',
      packages=['accuread'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
