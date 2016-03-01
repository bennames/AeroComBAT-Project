# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:20:33 2016

@author: Adam Zelenka
"""
import unittest

from Structures import Node



class NodeTest(unittest.TestCase):
    """Creates a test for the Node class.
    """
    def __init__(self):
        """Constructor declares Node object for testing.
        """
        self.myNode = None
    
    def setUp3(self):
        """Creates a fresh Node object with three defined coordinates
        """
        self.myNode = Node(0, [1.,2.,3.])
        
    def setUp2(self):
        """Creates a fresh Node object with two defined coordinates
        """
        self.myNode = Node(0, [1.,2.])
    
    def setUp1(self):
        """Creates a fresh Node object with one defined coordinate
        """
        self.myNode = Node(0, [1.])
        
    def setUp0(self):
        """Creates a fresh Node object with no defined coordinates
        """
        self.myNode = Node(0, [])
        
        
    def testLengths(self):
        """Tests to see if created Nodes have correctly defined coordinates
        """
        self.setUp0()
        self.assertEqual([0.,0.,0.], self.myNode.x)
        self.setUp1()
        self.assertEqual([1.,0.,0.], self.myNode.x)
        self.setUp2()
        self.assertEqual([1.,2.,0.], self.myNode.x)
        self.setUp3()
        self.assertEqual([1.,2.,3.], self.myNode.x)
    
    def testSetUpErrors(self):
        """Tests to see if the Node class will create a Node object with a 
           non-integer Node ID
        """
        while True:
            try:
                self.myNode = Node('a', [])
                break
            except TypeError as e:
                print(e)
                
        while True:
            try:
                self.myNode = Node(1, [1.,2.,3.,4.])
                break
            except ValueError as e:
                print(e)
                
if __name__ == '__main__': unittest.main()
                

           