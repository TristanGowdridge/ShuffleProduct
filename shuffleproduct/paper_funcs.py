# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:13:17 2023

@author: trist
"""


def term_to_latex(term):
    """
    For converting the array form of the generating series to symbolic form
    for latex. This only works for Duffing's equation and requires a1 = 1 and
    a2 = 100.
    """
    trow_str = ""
    brow_str = ""
    for i, (top, bot) in enumerate(term.T):
        if i != 0:
            trow_str += f"x_{top}"
        else:
            trow_str += f"{top} k_3"
        
        n_a2 = bot // 10
        n_a1 = bot - n_a2 * 10
        
        if n_a1 and not n_a2:
            if n_a1 == 1:
                brow_str += "a_1"
            else:
                brow_str += f"{n_a1}a_1"
            
        elif not n_a1 and n_a2:
            if n_a2 == 1:
                brow_str += "a_2"
            else:
                brow_str += f"{n_a2}a_2"
        
        elif n_a1 and n_a2:
            if n_a1 == 1:
                brow_str += "a_1+"
            else:
                brow_str += f"{n_a1}a_1+"
            
            if n_a2 == 1:
                brow_str += "a_2"
            else:
                brow_str += f"{n_a2}a_2"
        
        if i < term.shape[1]-1:
            trow_str += " & "
            brow_str += " & "
        else:
            trow_str += r" \\"
            brow_str += "0"
                
    print(
        r"\begin{bmatrix}"+f"\n\t{trow_str}\n\t{brow_str}\n"+r"\end{bmatrix}"
    )