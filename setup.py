import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='dsi6',  
     version='0.6',
     author="San Francisco dsi-cc6 cohort, shout out to Ritchie Kwan for providing the final code",
     author_email="mpopovich@gmail.com",
     description="Mostly metrics right now, by and for dsi students",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/MarkPopovich/dsi6",
     packages=['dsi6'] ,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )