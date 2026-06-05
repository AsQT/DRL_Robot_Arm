import os
import yaml
import sys
#   Add access if it is not in the system path.
if '../../../' + 'src' not in sys.path:
    sys.path.append('../../../' + 'src')

from config_loader import PROJECT_FOLDER_NAME

print(os.path.abspath(__file__))
CONST_PROJECT_FOLDER = os.getcwd().split('DRL_Robot_Manipulator')[0] + 'DRL_Robot_Manipulator'
print(CONST_PROJECT_FOLDER)