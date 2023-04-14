import numpy as np
import sympy as sym
import math

"""
Shuffle product has to fall into one of four categories, defined by the four 
if statements. These are defined by the lengths of the arrays in the 
argument of the shuffle product. 
The shuffle product acts on these arrays by reducing their size and then
applying a recursive definition, repeating the process until a halting
criteria is achieved.

mult1 is taken as the multiplier formed by the reduction of the LHS term and 
conversely, mult2 is taken from the reduction of the RHS term in the shuffle
product. Therefore mult1[0] should be taken from A and mult2[0] should be
taken from B.

All the information is stored in a dictionary, where each entry is a class.
the information stored in here is the array used to form the output, the depth
recursion (i.e., how many shuffle products deep that term is), the handedness
(whether the term was produced by the reduction of the LHS or the RHS, LHS is
represented by a 0 and the RHS is represented by a 1, the end term is
represented by an E), the path (this indicates the order of the operations to
get to this point i.e, 1001E would represent RHS -> LHS -> LHS -> RHS -> END,
the path is essentially the 'sum' of the previous handednesses), and the value
of the term.

this script is made up of one primary function named shuffle(), and then 
various subfunctions:
subshuffle() creates all the terms for the outputs, but provides no structure 
between the terms.
findparents() finds the previous terms used to make up the current term, this
is the function that determines finds the path taken by the term and writes
the path to the class in the dictionary. This provides structure between the
terms.
shufindex() uses the information from findparents() to stitch the terms
together in order, thus giving us the output.
collect() simply collects all the like terms in the list in the array format,
some issues with this listed below, but I don't think it should impact the
results in the way it is being used here.
multconcat() adds the multiplier to the beginning of the array.
collect() collects all the like terms in the list.
                                                              
                                                              
most variables have been hidden as they are not required as outputs, these are
only useful when debugging/ altering code. To retrieve the value, type global
and then the variable of interest just beneath the shuffle(A,B) line. This
*SHOULD* print the variable in the variable explorer.
"""



i = 0      
    
def shuffle(A, B, M): #A = first term, B = second term, M = multiplier
    global dictionary
    class shuffleclass():
        val = []
        array = []
        handedness = ''
        depth = []
        path = ''
    
    
    numofout = int((math.factorial(len(A[0])+len(B[0])-2))/(math.factorial(len(A[0])-1)*math.factorial(len(B[0])-1))) 
    
    
    dictionary = {}
    #-1th position is required for the terms with depth = 0, it has no meaning,
    #just a convenience thing.
    dictionary[-1] = shuffleclass()
    dictionary[-1].val = -1
    dictionary[-1].path = ''
    dictionary[-1].depth = -1
    
    depth = 0
    
    #collects all the like terms in the output, seems to be a glitch with this 
    #if entries are repeated in list i.e., l = [a,a,a,a,a,a,a,a,a] it adds to 
    #all copies of a, might be something to do with the way python stores 
    #memory. It works if different variable names are equivalent and then 
    #these make up the list, i.e, a=b=c=d=e=f then list = [a,b,c,d,e,f] but 
    #not when list = [a,a,a,a,a,a].
    
    def collect(O): #collects all the like terms in a list in the array format
        for i in range(len(O)-1):
            for k in range(i+1,len(O)):
                if not isinstance(O[k], str):
                    if not (isinstance(O[i], str) or isinstance(O[k], str)):
                        if (O[i][:,1:] == O[k][:,1:]).all() and O[i][1,0] == O[k][1,0]:
                            O[i][0,0] = O[k][0,0] + O[i][0,0]
                            O[k] = 'del'  
                    else:
                        continue
            
        length = len(O)
        j=0
        while j < length:
            if isinstance(O[j], str):
                del O[j]
                length = length - 1
                continue
            j=j+1
    
    def multconcat(M, O):
        multnew = np.array([[None, None],[None, None]])
        
        if len(M[0]) == 1:
            for osize in range(len(O)):
                O[osize][0,0] = M[0,0]*O[osize][0,0]
            return O
        if len(M[0]) == 2:
            for osize in range(len(O)):
                multnew[0,0] = M[0,0]*O[osize][0,0]
                multnew[0,1] = M[0,1]
                multnew[1,0] = M[1,0]
                multnew[1,1] = O[osize][1,0]
                O[osize] = np.delete(output[osize], 0, axis=1)
                O[osize] = np.hstack((multnew, O[osize]))
            return O
        if len(M[0]) >=3:
            for osize in range(len(O)):
                multnew = np.array([[None],[None]])
                multnew[0,0] = M[0,0]*O[osize][0,0]
                multnew[1,0] = M[1,0]
                O[osize][0,0] = M[0,len(M[0])-1]
                O[osize] = np.hstack((multnew,M[:, 1:len(M)],O[osize]))
            return O
    
    
    def shufindex(A, B, dictionary, numofout):
        pathlist = []
        for ipathlist in range(len(dictionary)-1):
            if len(dictionary[ipathlist].path) == len(A[0]) + len(B[0]) - 1:
                pathlist.append(dictionary[ipathlist].path)
        
        kout = 0
        outputindex = np.empty((numofout, 0)).tolist()
        for ioutind in pathlist:
            for joutind in np.arange(1,len(A[0])+len(B[0])):
                for koutind in range(len(dictionary)-1):
                    if dictionary[koutind].path == ioutind[:joutind]:
                        outputindex[kout].append(koutind)
            kout = kout + 1
                    
        output = np.empty((numofout, 0)).tolist()
        for iout in range(len(outputindex)):
            for jout in np.arange(len(outputindex[0])-1,-1, -1):  #done this way to reverse the order of the path, take [3, 2, 1, 0] rather than [0, 1, 2, 3]      
                output[iout].append(dictionary[(outputindex[iout][jout])].array)
    
        return output
            
    def findparent(dictionary, i):
        df = 1
        while True:
            if dictionary[i-df].depth == dictionary[i].depth - 1:
                dictionary[i].path = dictionary[i-df].path + dictionary[i].handedness
                break
            df = df + 1
    
    
    def subshuffle(A,B, dictionary, depth):
        global i
        #preallocating and rewriting for every iteration of the shuffle 
        #product, None is the only data type that seems to work with SymPy
        mult1 = np.array([[None],[None]]) 
        mult2 = np.array([[None],[None]])
        AA = None
        BB = None
        
        if len(A[0]) != 1 and len(B[0]) != 1:
            AA=A[:,:(len(A[0])-1)] #takes the first n-1 columns
            mult1[0] = A[0, len(A[0])-1]
            mult1[1] = A[1, len(A[0])-1] + B[1, len(B[0])-1]
            dictionary[i] = shuffleclass()
            dictionary[i].val = i
            dictionary[i].array = mult1
            dictionary[i].handedness = '0'
            dictionary[i].depth = depth
            findparent(dictionary, i)
            i = i + 1
            depth = depth + 1
            subshuffle(AA,B, dictionary, depth)
            depth = depth - 1
            
            BB=B[:,:(len(B[0])-1)]
            mult2[0] = B[0, len(B[0])-1]
            mult2[1] = B[1, len(B[0])-1] + A[1, len(A[0])-1]
            dictionary[i] = shuffleclass()
            dictionary[i].val = i
            dictionary[i].array = mult2
            dictionary[i].handedness = '1'
            dictionary[i].depth = depth
            findparent(dictionary, i)
            i = i + 1
            depth = depth + 1
            subshuffle(A,BB, dictionary, depth)
            depth = depth - 1
                
        if len(A[0]) == 1 and len(B[0]) != 1:
            BB=B[:,:(len(B[0])-1)]
            mult2[0] = B[0, len(B[0])-1]
            mult2[1] = A[1]+B[1, len(B[0])-1]
            dictionary[i] = shuffleclass()
            dictionary[i].val = i
            dictionary[i].array = mult2
            dictionary[i].handedness = '1'
            dictionary[i].depth = depth
            findparent(dictionary, i)
            i = i + 1
            depth = depth + 1
            subshuffle(A,BB, dictionary, depth)
            depth = depth - 1
            
        
        if len(A[0]) != 1 and len(B[0]) == 1:
            AA=A[:,:(len(A[0])-1)]
            mult1[0] = A[0, len(A[0])-1] 
            mult1[1] = A[1, len(A[0])-1] + B[1, 0]
            dictionary[i] = shuffleclass()
            dictionary[i].val = i
            dictionary[i].array = mult1
            dictionary[i].handedness = '0'
            dictionary[i].depth = depth
            findparent(dictionary, i)
            i = i + 1
            depth = depth + 1
            subshuffle(AA,B, dictionary, depth)
            depth = depth - 1  
            
            
        if len(A[0]) == 1 and len(B[0]) == 1:
            mult1[0] = A[0,0]*B[0,0]
            mult1[1] = A[1,0]+B[1,0]
            dictionary[i] = shuffleclass()
            dictionary[i].val = i
            dictionary[i].array = mult1
            dictionary[i].handedness = 'E'
            dictionary[i].depth = depth
            findparent(dictionary, i)
            i = i + 1
                
    subshuffle(A,B, dictionary, depth)
    output = shufindex(A, B, dictionary, numofout)
    
    for concat in range(len(output)):
        output[concat] = np.hstack(output[concat])
    collect(output)
    output = multconcat(M, output) 
    
    return output




        

# output1 = shuffle(g0,g0, multiplier)
def collect(O): #collects all the like terms in a list in the array format
    for i in range(len(O)-1):
        for k in range(i+1,len(O)):
            if not isinstance(O[k], str):
                if not (isinstance(O[i], str) or isinstance(O[k], str)):
                    if O[i].shape == O[k].shape:
                        if (O[i][:,1:] == O[k][:,1:]).all() and O[i][1,0] == O[k][1,0]:
                            O[i][0,0] = O[k][0,0] + O[i][0,0]
                            O[k] = 'del'  
                    else:               #else and continue might need changing, outputing results though.
                        continue
        
    length = len(O)
    j=0
    while j < length:
        if isinstance(O[j], str):
            del O[j]
            length = length - 1
            continue
        j=j+1
        





def shuff2(A, B, Mult):
    global i
    gg = {}
    
    
    adder = 0
    for AA in  A:
        for BB in  B:
            i=0
            gg[adder] = shuffle(AA, BB, Mult)
            adder = adder + 1
    
    ggout = [] #denests terms so all in same domain, easier to index
    for xx in range(len(gg)):
        ggout.extend(gg[xx])
   
    
    return ggout   




def shuff3(A, B, C, Mult):
    global i
    tempstore = {}
    multnull =  np.array([[1],[0]])
    adder = 0
    for AA in  A:
        for BB in  B:
            i=0
            tempstore[adder] = shuffle(AA, BB, multnull)
            adder = adder + 1
    
    tempstoreext = [] #denests terms so all in same domain, easier to index
    for xx in range(len(tempstore)):
        tempstoreext.extend(tempstore[xx])
    
    adder = 0
    ggg = {}
    for CC in  C:
        for GG in tempstoreext:
            i=0
            ggg[adder] = shuffle(CC, GG, Mult)
            adder = adder + 1
    
    gggout = [] #denests terms so all in same domain, easier to index
    
    for xx in range(len(ggg)):
        gggout.extend(ggg[xx])
    
     #get collect working, reduce number of terms after each iteration.
    collect(gggout)
    return gggout
    


a_1 = sym.Symbol('a')
a_2 = sym.Symbol('a_2')
x_0 = sym.Symbol('x_0')
x_1 = sym.Symbol('x_1')
vep_1 = sym.Symbol('vep_1')
vep_2 = sym.Symbol('vep_2')

multiplier1 = [np.array([[1, 1],[1, 0]])]
multiplier2 = [np.array([[1, 1],[1, 0]])]

identity = np.array([[1], [0]])


##############################################################################
################################# COMPUTING ##################################
##############################################################################


#Calculating g1
abcdefgh = shuff2(multiplier1, multiplier2, identity)







