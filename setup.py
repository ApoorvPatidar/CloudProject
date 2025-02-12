from setuptools import find_packages, setup
from typing import List

## Creating of the package which can also be added to the PyPi packages webiste 

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    
    ## -e. is used in the reqirements.txt file which indicates to the setup file ssuch that the setup files get built in the editable mode
    '''
    This function will return the list of the reqirements from the file 
    
    '''

    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if not  HYPHEN_E_DOT] # strip function removes any empty spaces and also removes any new line characters that is \n
        # if HYPHEN_E_DOT in requirements:
        #     requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
    name='cloudproject',
    version='0.1',
    author= 'Apoorv Patidar',
    email = 'apoorvpatidar.ap24@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt'))