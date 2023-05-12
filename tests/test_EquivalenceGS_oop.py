# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:56:07 2023

@author: trist
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()) + "\shuffleproduct")

import unittest
import numpy as np
import shuffle_oop as shfl

x0 = 0
x1 = 1

b = 1
a = 100


shuf_obj = shfl.BinaryShuffle()


class TestPaperQuadratic(unittest.TestCase):
    """
    This compares against the paper "An algebraic approach to nonlinear
    functional expansions"- Michel Fleiss. This paper is prefixed as 1) in
    the papers directory on GitHub. This only shows a binary shuffle, but goes
    quite deep on the iterations. I have subbed out the constants 'k1' for 'a'
    and 'k2' for 'b'.
    
    When running these tests, I've done a test for each term. For the manual
    checks, we neeed to calculate the lower order terms anyway. I've done this
    as if the higher order terms begin to diverge, we can still see if the
    lower order terms are correct, this in turn helps pinpoint where the errors
    may lie. Whereas, for the iterate_gs() function, this is not required as
    all are returned at once.
    """
    multiplier = np.array([
        [-b, x0],
        [ a,  0]
    ])
    
    g0 = shfl.GeneratingSeries(np.array([
        [ 1, x1],
        [ a,  0]
    ]))
    
    g1 = shfl.GeneratingSeries(np.array([
        [-2*b,   x0,   x1,  x1],
        [   a,  2*a,    a,   0]
    ]))
    
    g2 = [
        shfl.GeneratingSeries(np.array([
            [4*b**2,  x0, x1,  x0, x1, x1],
            [     a, 2*a,  a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [12*b**2, x0,  x0,  x1,  x1, x1],
            [     a, 2*a, 3*a, 2*a,   a,  0]
        ]))
    ]
    g2 = sorted(g2, key=hash)
    
    g3 = [
        shfl.GeneratingSeries(np.array([
            [  -8*b**3,  x0, x1,  x0, x1,  x0, x1, x1],
            [        a, 2*a,  a, 2*a,  a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ -24*b**3,  x0,  x0,  x1, x1, x0, x1, x1],
            [        a, 2*a, 3*a, 2*a,  a, 2*a, a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ -72*b**3, x0,   x0,  x1,  x0,  x1, x1, x1],
            [        a, 2*a, 3*a, 2*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ -24*b**3,  x0, x1,  x0,  x0,  x1, x1, x1],
            [        a, 2*a,  a, 2*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-144*b**3,  x0,  x0,  x0,  x1,  x1, x1, x1],
            [        a, 2*a, 3*a, 4*a, 3*a, 2*a,  a,  0]
        ]))
    ]
    g3 = sorted(g3, key=hash)
    
    g4 = [
        shfl.GeneratingSeries(np.array([
            [  48*b**4,  x0,  x0,  x1, x1,  x0, x1,  x0, x1, x1],
            [        a, 2*a, 3*a, 2*a,  a, 2*a,  a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ 144*b**4,  x0,  x0,  x1,  x0,  x1, x1,  x0, x1, x1],
            [        a, 2*a, 3*a, 2*a, 3*a, 2*a,  a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ 432*b**4,  x0,  x0,  x1,  x0,  x1,  x0,  x1, x1, x1],
            [        a, 2*a, 3*a, 2*a, 3*a, 2*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ 288*b**4,  x0,  x0,  x0,  x1,  x1, x1,  x0, x1, x1],
            [        a, 2*a, 3*a, 4*a, 3*a, 2*a,  a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [ 864*b**4,  x0,  x0,  x0,  x1,  x1,  x0,  x1, x1, x1],
            [        a, 2*a, 3*a, 4*a, 3*a, 2*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [1728*b**4,  x0,  x0,  x0,  x1,  x0,  x1,  x1, x1, x1],
            [        a, 2*a, 3*a, 4*a, 3*a, 4*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [864*b**4,  x0,  x0,  x1,  x0,  x0,  x1,  x1, x1, x1],
            [       a, 2*a, 3*a, 2*a, 3*a, 4*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [144*b**4,  x0,  x0,  x1, x1,  x0,  x0,  x1, x1, x1],
            [       a, 2*a, 3*a, 2*a,  a, 2*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [2880*b**4,  x0,  x0,  x0,  x0,  x1,  x1,  x1, x1, x1],
            [        a, 2*a, 3*a, 4*a, 5*a, 4*a, 3*a, 2*a,  a,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [16*b**4,  x0, x1,  x0, x1,  x0, x1,  x0, x1, x1],
            [      a, 2*a,  a, 2*a,  a, 2*a,  a, 2*a,  a,  0]
        ]))
    ]
    g4 = sorted(g4, key=hash)
    
    iter_args = (g0, multiplier, 2)
    
    def test_g1_manual(self):
        g1 = shuf_obj(TestPaperQuadratic.g0, TestPaperQuadratic.g0)
        
        assert len(g1) == 1
        assert isinstance(g1, list)
        
        g1_terms = []
        for g1_term in g1:
            g1_terms.append(g1_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        assert TestPaperQuadratic.g1.hard_equals(g1_terms[0])


    def test_g2_manual(self):
        g1 = shuf_obj(TestPaperQuadratic.g0, TestPaperQuadratic.g0)
        g1_terms = []
        for g1_term in g1:
            g1_terms.append(g1_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        g2 = []
        for g1_term in g1_terms:   
            g2.extend(shuf_obj(g1_term, TestPaperQuadratic.g0))
            g2.extend(shuf_obj(TestPaperQuadratic.g0, g1_term))
        assert (len(g2) == 4)
        
        g2 = shfl.collect(g2)
        
        assert (len(g2) == 2)
        
        g2_terms = []
        for g2_term in g2:
            g2_terms.append(g2_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        g2_terms = sorted(g2_terms, key=hash)
        for g2_a, g2_b in zip(g2_terms, TestPaperQuadratic.g2):
            assert g2_a.hard_equals(g2_b)
            
    
    def test_g3_manual(self):
        # Calculate the first shuffle. g0 with g0.
        g1 = shuf_obj(TestPaperQuadratic.g0, TestPaperQuadratic.g0)
        
        # Prepend the multiplier to each term.
        g1_terms = []
        for g1_term in g1:
            g1_terms.append(g1_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        g2 = []
        for g1_term in g1_terms:   
            g2.extend(shuf_obj(g1_term, TestPaperQuadratic.g0))
            g2.extend(shuf_obj(TestPaperQuadratic.g0, g1_term))
        g2 = shfl.collect(g2)
        
        g2_terms = []
        for g2_term in g2:
            g2_terms.append(g2_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        
        g2_terms = sorted(g2_terms, key=hash)

        for g2_a, g2_b in zip(g2_terms, TestPaperQuadratic.g2):
            assert g2_a.hard_equals(g2_b)

        g3 = []
        for g2_term in g2_terms:
            g3.extend(shuf_obj(g2_term, TestPaperQuadratic.g0))
            g3.extend(shuf_obj(TestPaperQuadratic.g0, g2_term))
        g3.extend(shuf_obj(g1_terms[0], g1_terms[0]))
        g3 = shfl.collect(g3)

        g3_terms = []
        for g3_term in g3:
            temp = g3_term.prepend_multiplier(TestPaperQuadratic.multiplier)
            g3_terms.append(temp)

        g3_terms = sorted(g3_terms, key=hash)
    
        for g3_a, g3_b in zip(g3_terms, TestPaperQuadratic.g3):
            assert g3_a.hard_equals(g3_b) 
        
        
        
    def test_g1_nShuffles(self):
        g1 = shfl.nShuffles(TestPaperQuadratic.g0, TestPaperQuadratic.g0)
        g1_terms = []
        for g1_term in g1:
            g1_terms.append(g1_term.prepend_multiplier(TestPaperQuadratic.multiplier))
            
        assert TestPaperQuadratic.g1.hard_equals(g1_terms[0])
        

    def test_g2_nShuffles(self):
        g1 = shfl.nShuffles(TestPaperQuadratic.g0, TestPaperQuadratic.g0)
        g1_terms = []
        for g1_term in g1:
            g1_terms.append(g1_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        g2 = []
        for gs_term in g1_terms:   
            g2.extend(shfl.nShuffles(gs_term, TestPaperQuadratic.g0))
            g2.extend(shfl.nShuffles(TestPaperQuadratic.g0, gs_term))
        g2 = shfl.collect(g2)
        
        assert(len(g2) == 2)
        
        g2_terms = []
        for g2_term in g2:
            g2_terms.append(g2_term.prepend_multiplier(TestPaperQuadratic.multiplier))
        
        g2_terms = sorted(g2_terms, key=hash)
        for g2_a, g2_b in zip(g2_terms, TestPaperQuadratic.g2):
            assert g2_a.hard_equals(g2_b) 
    
    
    def test_iterate_gs1(self):
        gs_term = 1
        all_gs = shfl.iterate_gs(*TestPaperQuadratic.iter_args, iter_depth=gs_term,
                        return_type=dict)
        g1 = all_gs[gs_term][0]
        
        assert g1.hard_equals(TestPaperQuadratic.g1)


    def test_iterate_gs2(self):
        gs_term = 2
        all_gs = shfl.iterate_gs(*TestPaperQuadratic.iter_args, iter_depth=gs_term,
                        return_type=dict)
        g2 = all_gs[gs_term]
        
        g2_terms = sorted(g2, key=hash)
        for g2_a, g2_b in zip(g2_terms, TestPaperQuadratic.g2):
            assert g2_a.hard_equals(g2_b)


    def test_iterate_gs3(self):
        gs_term = 3
        all_gs = shfl.iterate_gs(*TestPaperQuadratic.iter_args, iter_depth=gs_term,
                        return_type=dict)
        g3 = all_gs[gs_term]
        
        g3_terms = sorted(g3, key=hash)
        for g3_a, g3_b in zip(g3_terms, TestPaperQuadratic.g3):
            assert g3_a.hard_equals(g3_b)


    def test_iterate_gs4(self):
        """
        Since the papers only list a subset of the fourth order terms, we need
        to take a count and assert that it's equal to the length.
        """
        gs_term = 4
        all_gs = shfl.iterate_gs(*TestPaperQuadratic.iter_args, iter_depth=gs_term,
                        return_type=dict)
        g4 = all_gs[gs_term]
        
        number_of_matches = 0
        for i in g4:
            for j in TestPaperQuadratic.g4:
                if i.hard_equals(j):
                    number_of_matches += 1
        assert (number_of_matches == len(TestPaperQuadratic.g4))


    def test_iterate_gs1to4(self):
        """
        Included this just incase later terms are modifying the previous terms.
        """
        all_gs = shfl.iterate_gs(*TestPaperQuadratic.iter_args, iter_depth=4,
                        return_type=dict)

        g1 = all_gs[1][0]
        assert g1.hard_equals(TestPaperQuadratic.g1)
        
        g2_terms = sorted(all_gs[2], key=hash)
        for g2_a, g2_b in zip(g2_terms, TestPaperQuadratic.g2):
            assert g2_a.hard_equals(g2_b)
        
        g3_terms = sorted(all_gs[3], key=hash)
        for g3_a, g3_b in zip(g3_terms, TestPaperQuadratic.g3):
            assert g3_a.hard_equals(g3_b)
        
        number_of_matches4 = 0
        for i in all_gs[4]:
            for j in TestPaperQuadratic.g4:
                if i.hard_equals(j):
                    number_of_matches4 += 1
        assert (number_of_matches4 == len(TestPaperQuadratic.g4))



class TestPaperImp(unittest.TestCase):
    """
    Compares the GS terms to the ones obtained in "Functional Analysis of
    Nonlinear Circuits- A Generating Power Series Approach" - M. Lamnabhi. The
    system (EQ 21) they use contains a cubic nonlinearity and two multipliers.
    The integral equation is of the form:
        \int(i)dt + i - (1/3)*i**3 = e_s
    
    This is a somewhat simple example, but it demonstrates the cubic shuffle, 
    and other papers only make use of one multiplier.
    """
    multipliers = [
        np.array([
            [1/3],
            [  0]
        ]),
        np.array([
            [-1/3, x0],
            [   1,  0]
        ]),
    ]

    g0 = shfl.GeneratingSeries(np.array([
        [1, x1],
        [1,  0]
    ]))
    
    iter_args = (g0, multipliers, 3)
    
    gs_unsorted = [
        shfl.GeneratingSeries(np.array([
            [1, x1],
            [1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [2, x1, x1, x1],
            [3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-2, x0, x1, x1, x1],
            [ 1,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [40, x1, x1, x1, x1, x1],
            [ 5,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-40, x0, x1, x1, x1, x1, x1],
            [  3,  5,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-16, x1, x0, x1, x1, x1, x1],
            [  3,  2,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-4, x1, x1, x0, x1, x1, x1],
            [ 3,  2,  1,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [-40, x0, x1, x1, x1, x1, x1],
            [  1,  5,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [40, x0, x0, x1, x1, x1, x1, x1],
            [ 1,  3,  5,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [16, x0, x1, x0, x1, x1, x1, x1],
            [ 1,  3,  2,  4,  3,  2,  1,  0]
        ])),
        shfl.GeneratingSeries(np.array([
            [4, x0, x1, x1, x0, x1, x1, x1],
            [1,  3,  2,  1,  3,  2,  1,  0]
        ]))       
    ]
    correct_gs = sorted(gs_unsorted, key=hash)
    
    
    def test_iterate_gs_order2(self):
        scheme = shfl.iterate_gs(*TestPaperImp.iter_args, iter_depth=2, return_type=list)
        scheme = sorted(scheme, key=hash)
        assert (scheme == TestPaperImp.correct_gs)
        
        


# class TestPaperCubic(unittest.TestCase):
#     """
#     Duffing paper. This now incorporates a y ** 3, meaning two shuffles are 
#     required.
#     """
#     g0 = [
#         shfl.GeneratingSeries(np.array([
#             [  1,  x0, x1],
#             [-a1, -a2,  0]])),
#         ]
#     def test_g1_manual(self):
#         print("Need to do cubic case.")
#         assert True

#     def test_g2_manual(self):
#         print("Need to do cubic case.")
#         assert True
        
#     def test_g3_manual(self):
#         print("Need to do cubic case.")
#         assert True
    
#     def test_g1_nShuffles(self):
#         print("Need to do cubic case.")
#         assert True

#     def test_g2_nShuffles(self):
#         print("Need to do cubic case.")
#         assert True
        
#     def test_g3_nShuffles(self):
#         print("Need to do cubic case.")
#         assert True
                    
#     def test_g1_iterate(self):
#         print("Need to do cubic case.")
#         assert True
    
#     def test_g2_iterate(self):
#         print("Need to do cubic case.")
#         assert True
    
#     def test_g3_iterate(self):
#         print("Need to do cubic case.")
#         assert True
        
#     def test_g1to3_iterate(self):
#         print("Need to do cubic case.")
#         assert True        

        
if __name__ == "__main__":
    unittest.main()