import sys
import os

import ast

from cmd import Cmd
from threading import Thread

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../AeroComBAT/'))

from AeroComBAT.FEM import Model

class UserInterface(Cmd):
    def __init__(self):
        super(UserInterface, self).__init__()
        self.Model = Model()
        
    def do_add_material(self, args):
        """Adds a material to the AeroComBAT model.\n\n
        Must be in the form of:\n [MID, mat_name, mat_type, mat_constants,mat_t]\n\n
        Supported 'mat_type's are: iso, trans_iso, and ortho"""
        args = ast.literal_eval(args)
        MID = args[0]
        mat_name = args[1]
        mat_type = args[2]
        mat_constants = args[3]
        mat_t = args[4]
        print(args)
        self.Model.materials.addMat(MID, mat_name, mat_type,\
            mat_constants,mat_t)
    def do_print_materials(self, args):
        """Prints a list of materials contained in the model."""
        self.Model.materials.printSummary()

    def do_quit(self, args):
        """Quits the program."""
        print("Quitting.")
        raise SystemExit
    
#    def do_fullModel(self,args):
#        """Opens up a window displaying the """
#        t = Thread(target=display)
#        t.daemon = True
#        t.start()

if __name__ == '__main__':
    prompt = UserInterface()
    
    prompt.prompt = '>>> '
    print('####################################################################')
    print('##   ___                 _____                ______  ___ _____  ###')
    print('##  / _ \               /  __ \               | ___ \/ _ \_   _| ###')
    print('## / /_\ \ ___ _ __ ___ | /  \/ ___  _ __ ___ | |_/ / /_\ \| |   ###')
    print('## |  _  |/ _ \ `__/ _ \| |    / _ \| `_ ` _ \| ___ \  _  || |   ###')
    print('## | | | |  __/ | | (_) | \__/\ (_) | | | | | | |_/ / | | || |   ###')
    print('## \_| |_/\___|_|  \___/ \____/\___/|_| |_| |_\____/\_| |_/\_/   ###')
    print('####################################################################\n\n')
    print('Welcome to the AeroComBAT interface!')
    print('====================================')
    print('To list the possible commands, type "?"')
    prompt.cmdloop()