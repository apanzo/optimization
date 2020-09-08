installation using command line:

- *navigate to desired installation directory*
- **git clone https://github.com/apanzo/optimization** # get the project
- **cd optimization**                  # navigate to project
- *if installing dependencies inside virtual environment, proceed from step 2 in virtual environment setup guide*
- **setup.bat**                        # install dependencies
- **app\\run.py**                      # run the framework

virtual environment setup guide in cmd:

- *navigate to desired virtual environment directory (can be any)*
- **python -m venv venv**              # make a virtual environment
- **venv\\scripts\\activate.bat**        # activate virtual environment
- **cd (project_path)\\optimization**   # navigate to project, if not yet done
- **setup.bat**                        # install required packages
- **app\\run.py**                      # run the framework
- **deactivate**                       # escape the virtual environment

documentation can be built using:

- **cd (project_path)\\optimization\\docs**         # navigate to docs folder
- **make html**                        # build html documentation
- *the program will be run during the build, press* **Enter** *when Ended is print*
- **build\\html\\index.html**            # open the html documentation

readme.html is made using:

- **rst2html README.rst readme.html**