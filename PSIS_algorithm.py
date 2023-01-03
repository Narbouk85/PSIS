#env python3.10

"""
@author: Nampoina Ravelomanana
"""
import numpy as np




def initialize(img):
    """
    """
    xi, yi, zi  = img.shape
    N = 2*xi*yi -xi - yi

    img_ind = np.array(range(xi*yi)).reshape((xi,yi))
    p0 = np.concatenate((img_ind[:-1,:].reshape(-1) , img_ind[:,:-1].reshape(-1)))
    p1 = np.concatenate((img_ind[1:,:].reshape(-1), img_ind[:,1:].reshape(-1)))

    img = img.reshape((xi*yi, zi))

    return N, p0, p1, np.max(abs(img[p0,:] - img[p1,:]), axis = 1), img



def bounded(R, Q, g, deltaPrime):
    #R is the considered region, a vector of pixels
    l = len(R)
    if (l!= 0):

        b = g* np.sqrt((1/(2*Q*l))* (np.log2(2/deltaPrime) + np.log2(((l + 2)**(min(l,g))) / (l+1))))

        return b

    else:
        return 0

def merging_predicate(R1, R2, Q, g, deltaPrime):
    b1 = bounded(R1, Q, g, deltaPrime)
    b2 = bounded(R2, Q, g, deltaPrime)

    return  np.all(np.abs(np.mean(R1, axis=0) - np.mean(R2, axis=0)) <= b1+b2)

def PSIS(img, Q, g, deltaPrime):
    N, p0, p1, f, img = initialize(img)

    ind_sort = np.argsort(f)
    f = f[ind_sort]
    p1 = p1[ind_sort]
    p0 = p0[ind_sort]

    Reg = [0]*N
    Pixel_Labels = -np.ones(img.shape[0])

    nlabel = 0
    for n in range(N):


        # if pixel is not yet labeled then we give it a label as nlabel
        if (Pixel_Labels[p0[n]]==-1):

            Pixel_Labels[p0[n]] = nlabel

            Reg[nlabel] = img[p0[n],:]
            nlabel +=1
        if (Pixel_Labels[p1[n]]==-1):
            Pixel_Labels[p1[n]] = nlabel

            Reg[nlabel] = img[p1[n],:]
            nlabel +=1
        # check if two regions can be merged
        if (Pixel_Labels[p0[n]] != Pixel_Labels[p1[n]]):

            r = merging_predicate(Reg[int(Pixel_Labels[p0[n]])], Reg[int(Pixel_Labels[p1[n]])], Q, g, deltaPrime)
            if r :
                label_merge = min(Pixel_Labels[p0[n]], Pixel_Labels[p1[n]])
                label_remove = max(Pixel_Labels[p0[n]], Pixel_Labels[p1[n]])
                Pixel_Labels[Pixel_Labels==label_remove] = label_merge
                Reg[int(label_merge)] = np.concatenate((Reg[int(label_merge)], Reg[int(label_remove)]))
                Reg[int(label_remove)] = []


    return Pixel_Labels

