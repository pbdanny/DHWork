import pandas as pd
import numpy as np
import math
import operator
from itertools import compress


def PCA(corr_mat, n_Components):
    n = (n_Components)
    U, S, V = np.linalg.svd(corr_mat, full_matrices=False)
    propexp_ = S / (S.sum())
    eig_pairs = []
    l = U.shape[0]
    for i in range(n):
        idx, Value = max(enumerate(S), key=operator.itemgetter(1))
        if i == 0:
            eig_pairs = [[Value], U[:, idx].reshape(l, 1), [propexp_[idx]]]
        else:
            eig_pairs[0] += [Value]
            eig_pairs[1] = np.hstack((eig_pairs[1], U[:, idx].reshape(l, 1)))
            eig_pairs[2] += [propexp_[idx]]
        S = np.delete(S, idx)
        U = np.delete(U, idx, 1)
        propexp_ = np.delete(propexp_, idx)
    del U, S, V
    return eig_pairs
    # Function to rotate the vectors obliquely (also does varimax)


def ortho_rotation(lam, method='varimax', eps=1e-6, itermax=100):
    if (method == 'varimax'):
        gamma = 1.0
    if (method == 'quartimax'):
        gamma = 0.0
    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0
    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new
    return np.dot(lam, R)


# scan through the squared corrs and identify the cluster which the variable is most associated with.
def ClusterNum(ClusCorr):
    for i in ClusCorr:
        if i[0] ** 2 >= i[1] ** 2:
            yield True
        else:
            yield False


def listBuildr(ClusCorr, VarIndx):
    i = 0
    left = []
    right = []
    for item in ClusterNum(ClusCorr):
        if item == True:
            left += [VarIndx[i]]
        else:
            right += [VarIndx[i]]
        i += 1
    return left, right


def genVal(x, valType):
    for i in range(x):
        if valType == 'zero':
            yield 0
        elif valType == 'cent':
            yield 1 / math.sqrt(x)


def evecs(Corr_mat, varIdx, Method, ncomps, asType):
    temp_mat = Corr_mat[varIdx, :][:, varIdx]
    if Method == 'PCA':
        Comp = PCA(temp_mat, ncomps)[1]
        if asType == 'list':
            return list(Comp)
        if asType == 'array':
            return Comp
    if Method == 'Centroid':  # add an exception here to fail when ncomps > 1
        Comp = [i for i in genVal(len(varIdx), 'cent')]
        if asType == 'list':
            return Comp
        if asType == 'array':
            return np.array(Comp).reshape(len(varIdx), 1)


def createComp(Corr_mat, tup, Method):
    list1 = evecs(Corr_mat, tup[0], Method, 1, 'list') + [i for i in genVal(len(tup[1]), 'zero')]
    list2 = [i for i in genVal(len(tup[0]), 'zero')] + evecs(Corr_mat, tup[1], Method, 1, 'list')
    list_ = zip(list1, list2)
    return np.array(list_)


def varExp(Corr_mat, varIdx, Method, ncomps, comps=None):
    if comps is not None:
        Evec = comps
    else:
        Evec = evecs(Corr_mat, varIdx, Method, ncomps, 'array')
    temp_mat = Corr_mat[varIdx, :][:, varIdx]
    if ncomps == 1:
        VarianceExp = np.multiply(temp_mat, np.dot(Evec, Evec.T)).sum()
    elif ncomps > 1:
        list_ = [np.multiply(temp_mat, np.dot(Evec[:, x:x + 1], Evec[:, x:x + 1].T)).sum() for x in range(ncomps)]
        VarianceExp = np.array(list_).reshape(1, ncomps)
    return VarianceExp


def SpecificCorr(Corr_mat, VarIdx=[], Split=None, Method='PCA', nComps=None, comp_=None):
    if Split is not None:
        VarIdx = Split[0] + Split[1]
    temp_mat = Corr_mat[VarIdx, :][:, VarIdx]
    if comp_ is None:
        comp_ = createComp(Corr_mat, Split, Method)
    Cov = np.dot(temp_mat, comp_)
    Corr = np.divide(Cov, np.sqrt(varExp(Corr_mat, VarIdx, Method, nComps, comp_)))
    return Corr


def varExpAll(Corr_mat, Split, Method):
    return varExp(Corr_mat, Split[0], Method, 1), varExp(Corr_mat, Split[1], Method, 1)


def BinarySplit(Corr_mat, VarIdx):
    temp_Mat = Corr_mat[VarIdx, :][:, VarIdx]
    Raw_loadings = PCA(temp_Mat, 2)[1]
    Rot_loadings = ortho_rotation(Raw_loadings, method='quartimax', eps=1e-8, itermax=100)
    Corr = SpecificCorr(Corr_mat, VarIdx, Split=None, Method='PCA', nComps=2, comp_=Rot_loadings)
    return listBuildr(Corr, VarIdx)


# NCS Phase
def NCS(Corr_mat, Bsplt, Maxiter, Method):
    k = 0
    Converged = False
    Terminal = False
    while Terminal == False:
        if k > 0:
            Bsplt = Nsplt
        Corr = SpecificCorr(Corr_mat, VarIdx=[], Split=Bsplt, Method=Method, nComps=2, comp_=None)
        VarIdx = Bsplt[0] + Bsplt[1]
        Nsplt = listBuildr(Corr, VarIdx)
        temp_ = varExpAll(Corr_mat, Nsplt, Method)
        if Nsplt == Bsplt:
            Converged = True
        else:
            k += 1
        Terminal = (k == Maxiter) or (Converged == True)
    return Nsplt


# Search Phase
def RsqRatio(Corr_mat, Nsplt, Method):
    VarIdx = Nsplt[0] + Nsplt[1]
    Corr = SpecificCorr(Corr_mat, VarIdx=[], Split=Nsplt, Method=Method, nComps=2, comp_=None)
    RsqRatio = []
    i = 0
    for item in Corr:
        if VarIdx[i] in Nsplt[0]:
            RsqRatio += [((1 - item[0] ** 2) / (1 - item[1] ** 2), VarIdx[i])]
        else:
            RsqRatio += [((1 - item[1] ** 2) / (1 - item[0] ** 2), VarIdx[i])]
        i += 1
    return RsqRatio


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def changeSplt(newList, tracker, x):
    for item in newList:
        if x in item:
            item.remove(x)
        else:
            item += [x]
    tracker = f7(tracker)
    tracker += [x]
    return newList, tracker


def mapChange(Rsq, newList, tracker, threshold):
    Max_ = max(Rsq)
    if Max_[1] >= threshold:
        if len(tracker) != sum((len(newList[0]), len(newList[1]))):
            if Max_[1] not in tracker:
                newList, tracker = changeSplt(newList, tracker, Max_[1])
                return newList, Max_[1], tracker
            elif Max_[1] in tracker:
                Rsq.remove(Max_)
                return mapChange(Rsq, newList, tracker, threshold)
        else:
            return newList, 'Surrrrrender the booty!', tracker
    else:
        return newList, 'Land hoooo!', tracker


def searchPhase(Corr_mat, Nsplt, Method, MaxSearch, tracker=[], counter_=0):
    print('All hand hoay!')
    while counter_ < MaxSearch:
        newList = (Nsplt[0][:], Nsplt[1][:])
        oldVariance = varExpAll(Corr_mat, Nsplt, Method)
        propExp_ = (oldVariance[0] / len(Nsplt[0])), (oldVariance[1] / len(Nsplt[1]))
        Rsq = RsqRatio(Corr_mat, newList, Method)
        newList, idx, tracker = mapChange(Rsq, newList, tracker, threshold=0)
        if idx == 'Land hoooo!':
            tracker = []
            counter_ += 1
        elif idx == 'Surrrrrender the booty!':
            tracker = []
            counter_ += 1
        else:
            newVariance = varExpAll(Corr_mat, newList, Method)
            if sum(oldVariance) < sum(newVariance):
                comp_ = createComp(Corr_mat, newList, Method)
                list_ = newList[0] + newList[1]
                corrCol = Corr_mat[idx, list_]
                Corr = np.dot(corrCol, comp_)
                Left = (idx in Nsplt[0])
                if (Left == True and (Corr[0] ** 2 < Corr[1] ** 2)) or (
                        Left == False and (Corr[0] ** 2 > Corr[1] ** 2)):
                    #                     print 'Blow the man down!!!'
                    Nsplt = newList
                    propExp_ = (newVariance[0] / len(Nsplt[0])), (newVariance[1] / len(Nsplt[1]))
    print('Scuttle')
    return Nsplt, propExp_


# Defining a Node class to store the cluster node
class Node(object):
    def __init__(self, Varlist, Variance):
        self.Cluster = (Varlist, Variance)

    def Variablelist(self):
        return self.Cluster[0]

    def Criterion(self):
        if len(self.Cluster[0]) <= 2:
            return 13245
        else:
            return self.Cluster[1]


# A Nodelist class is defined and an instance of this class is a list.
# More precicely a list of Node Class instances.
class Nodelist(Node):
    def __init__(self, Clusters=None):
        if not Clusters:
            self.Clusters = []
        else:
            self.Clusters = Clusters

    def add(self, Cluster):
        if isinstance(Cluster, Node) and len(Cluster.Variablelist()) > 0:
            self.Clusters.append(Cluster)
        return self

    def Crit(self):
        return (x.Criterion() for x in self.Clusters)

    def pop(self, method):
        if method == 'PCA':
            idx, value = max(enumerate(self.Crit()), key=operator.itemgetter(1))
        if method == 'Centroid':
            idx, value = min(enumerate(self.Crit()), key=operator.itemgetter(1))
        return self.Clusters.pop(idx)

    def ClusLen(self):
        return len(self.Clusters)

    def NthItem(self, n):
        return self.Clusters[n]


def splitting(Corr_mat, Method, Maxiter, MaxClus, MaxSearch):
    VarIdx = range(Corr_mat.shape[1])
    VC = Nodelist()
    loopnum = 0
    NumEigGrt1 = 9999
    print(varExp(Corr_mat, VarIdx, Method, 1))
    VC.add(Node(VarIdx, varExp(Corr_mat, VarIdx, Method, 1)))
    while (VC.ClusLen() < MaxClus) or (NumEigGrt1 == 0):
        print(loopnum)
        PopedNode = VC.pop(Method)
        VarIdx = PopedNode.Variablelist()
        Bsplt = BinarySplit(Corr_mat, VarIdx)
        del PopedNode
        Nsplt = NCS(Corr_mat, Bsplt, Maxiter, Method)
        finalNodes = searchPhase(Corr_mat, Nsplt, Method, MaxSearch)
        print('finalNodes:', len(finalNodes[0][0]), len(finalNodes[0][1]))
        print('varianceExp:', varExpAll(Corr_mat, (finalNodes[0][0], finalNodes[0][1]), Method))
        VC.add(Node(finalNodes[0][0], finalNodes[1][0]))
        VC.add(Node(finalNodes[0][1], finalNodes[1][1]))
        terminals = 0
        for NodeItem in VC.Crit():
            totVar = 0
            if (Method == 'Centroid' and NodeItem >= .75) or (Method == 'PCA' and NodeItem >= 1):
                terminals += 1
            totVar += NodeItem
        loopnum += 1
    return [[j, VC.NthItem(j).Variablelist()] for j in range(VC.ClusLen())]


def pushout(Clusters, VarNames, Outpath):
    OutVarClus = {'Cluster_Num': [], 'feature_name': []}
    rownum = 0
    OutVarClus = pd.DataFrame(OutVarClus)
    for i in range(len(Clusters)):
        item = Clusters[i][1]
        for j in range(len(item)):
            OutVarClus.loc[rownum] = [(i + 1), VarNames[item[j]]]
            rownum += 1
    if Outpath is not None:
        OutVarClus.to_csv(Outpath, index=False)
        print('Have fun going through CSVs now >.< !!!')

    return OutVarClus


def VarClus(Corr_mat, VarNames, Outpath, MaxClus, Method='Centroid', Maxiter=5, MaxSearch=10):
    """
        Function Name: VarClus(Corr_mat, VarNames, Outpath, MaxClus, Method= 'Centroid', Maxiter= 3, MaxSearch= 3)
        Use: Build mutually exclusive clusters of variables
        Corr_mat = Correlation matrix of input variables
        VarNames = provide list of names for variables used to create the correlation matrix in the order of the their positions
        Outpath = 'user/home/varclusrout.csv'
        MaxClus = number of cluster need (currently only heiracrhical so produces clusters = maxclus eg: 60 to produce 60 clusters)
        Method= 'Centroid' (use either 'Centroid' if you need unweighted linear combinations else 'PCA')
        Maxiter= number of iterations in Nearest component sorting. Use 3 default for Centriod and 5 default for PCA
        MaxSearch= number of iterations in Search Phasr. Use 3 default for Centriod and 10 default for PCA
    """
    Out = splitting(Corr_mat, Method, Maxiter, MaxClus, MaxSearch)
    return pushout(Out, VarNames, Outpath)
