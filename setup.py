import setuptools

setuptools.setup(name='tklpy',
      version='0.0.1',
      description='Python implementation of the Transfer Kernel Learning method developed by Long et. al. (2015).',
      url='http://github.com/jrjoaorenato/tklpy',
      download_url = 'https://github.com/jrjoaorenato/tklpy/archive/0.0.1.tar.gz',
      author='João Renato Ribeiro Manesco',
      author_email='joaorenatorm@gmail.com',
      license='MIT',
      packages=['tklpy'],
      keywords = ['Domain Adaptation', 'Machine Learning', 'TKL', 'Domain Invariant Transfer Kernel Learning'],
      install_requires=[            # I get to this in a second
            'numpy',
            'qpsolvers',
            'quadprog',
            'scikit-learn',
            'scipy'
      ],
      classifiers=[
            'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Developers',      # Define that your audience are developers
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: MIT License',   # Again, pick a license
            'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
            'Programming Language :: Python :: 3.8',
      ],
      )