##__init__.py is used to mark a folder as package
from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirnments=[]
    with open(file_path) as file_obj:
        requirnments=file_obj.readlines()
        ##requirnments.tx\n so ve neeed to remove \n
        requirnments=[req.replace("\n","") for req in requirnments]
        ##-e . is used to connect requirnments.txt to setup.py
        if HYPEN_E_DOT in requirnments:
            requirnments.remove(HYPEN_E_DOT)
    return requirnments
setup(
    name='abc',
    version='0.0.1',
    author='abc',
    author_email='abc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
