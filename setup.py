from setuptools import setup

setup(
    name='idyompy',
    version='0.1.0',    
    description='IDyOMpy: a New Python Implementation for IDyOM, a Statistical Model of Musical Expectations',
    url='https://github.com/GuiMarion/IDyOMpy',
    author='Guilhem Marion',
    author_email='gmarionfr@gmail.com',
    license='GNU GPL',
    packages=['idyompy'],
    install_requires=['matplotlib==3.7.2',
                        'mido==1.3.0',
                        'numpy==1.25.2',
                        'pretty_midi==0.2.10',
                        'scipy==1.11.1',
                        'setuptools==68.0.0',
                        'setuptools==67.6.1',
                        'tqdm==4.65.0'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL License',  
        'Operating System :: POSIX :: MacOs',        
        'Programming Language :: Python :: 3.11',
    ],
)

