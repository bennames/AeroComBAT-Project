# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 22:20:33 2016

@author: Adam Zelenka
"""
import unittest

from Structures import Node, Material


class NodeTest(unittest.TestCase):
    
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
        
    def tearDown(self):
        self.myNode.dispose()
        self.myNode = None
        
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
        self.tearDown()
    
    def testSetUpErrors(self):
        """Tests to see if the Node class will create a Node object with a 
           non-integer Node ID
        """
        try:
            self.myNode = Node('a', [])
        except TypeError as e:
            print(e)
                
        try:
            self.myNode = Node(1, [1.,2.,3.,4.])
        except ValueError as e:
            print(e)
        self.tearDown()
                

    
class MaterialTest(unittest.TestCase):
    
    def setUp(self):
        self.myMaterial = Material(1,'iron','iso',[210.,0.2,7850.],0.1,{})
        
    def tearDown(self):
        self.myMaterial.dispose()
        self.myMaterial = None
    
    def testMaterial(self):
        self.setUp()
        self.assertEqual(220.,self.myMaterial.E1)

    
if __name__ == '__main__':
    test_classes_to_run = [NodeTest, MaterialTest]

    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    print(suites_list)
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
           