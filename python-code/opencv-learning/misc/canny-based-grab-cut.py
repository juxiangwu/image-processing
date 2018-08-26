#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import cv2

import sklearn
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import igraph
import math

import directories
from visualization import *
import settings
from scoreOutput import threshold

ImagesAndBBoxes = directories.loadImagesAndBBoxes()
directories.ensureDir(directories.output)

def gaussianBlur(arr, sigma=3):
        return scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='reflect', cval=0.0)

def minCuts(vertices, edges, capacities):
        assert len(edges) == len(capacities)
        assert min(capacities) >= 0

        #Now I need to make the graph
        g = igraph.Graph(directed=False)

        g.add_vertices(map(str, vertices))
        g.add_vertices(["source", "sink"])

        g.add_edges([map(str, edge) for edge in edges])

        #igraph.plot(g.as_undirected(), layout="fr", vertex_label=None, edge_width=[2 * cap for cap in capacityList])

        capacities = [int(cap * 1000) for cap in capacities]

        print("Beginning cut...")
        return g.as_directed().all_st_mincuts("source", "sink", capacity=capacities * 2)

def normalizeList(l, shift=False):
        l = np.asarray([l], dtype='float')

        if shift:
                l -= np.average(l)

        l = sklearn.preprocessing.normalize(l, norm='l1', axis=1, copy=True)
        l = list(l[0])
        return [val * 100 for val in l]

def normalizeArr(arr, shift=False):
        return np.reshape(np.asarray(normalizeList(arr.flatten(), shift)), arr.shape)

def initBinaryEdges(img):
        edges = np.zeros(img.shape[:2], dtype="float")

        for apertureSize in settings.apertureSizes:
                for minEdgeStrength in range(0, 500, 50):
                        e = cv2.Canny(image=img, threshold1=2*minEdgeStrength, threshold2=minEdgeStrength, L2gradient=True, apertureSize=apertureSize)
                        #print(minEdgeStrength)
                        #visualize(e)
                        e = normalizeArr(e)
                        edges += e

                #visualize(edges)

        edges = edges ** 4
        #edges = gaussianBlur(edges, sigma=1) #I'd like to try this, but I can't because it makes mincut run too slowly
        normalizeArr(edges)

        print("Creating binary edges...")
        verticalEdges = [[(y, x), (y + 1, x)] for x in range(img.shape[1]) for y in range(img.shape[0] - 1)]
        horisontalEdges = [[(y, x), (y, x + 1)] for x in range(img.shape[1] - 1) for y in range(img.shape[0])]
        binaryEdges = horisontalEdges + verticalEdges

        if settings.diagonal:
                de = [[(y, x), (y + 1, x + 1)] for x in range(img.shape[1] - 1) for y in range(img.shape[0] - 1)]
                de2 = [[(y, x), (y + 1, x - 1)] for x in range(1, img.shape[1]) for y in range(img.shape[0] - 1)]
                binaryEdges += de + de2

        binaryCapacities = []
        for pts in binaryEdges:
                cap = 1/(edges[pts[0]]+edges[pts[1]] + .001)

                assert cap >= 0
                assert not math.isnan(cap)

                binaryCapacities.append(cap)

        normalizedCaps = [cap * settings.binaryEdgeStrength for cap in normalizeList(binaryCapacities)]
        print(settings.binaryEdgeStrength)

        assert min(normalizedCaps) >= 0

        return binaryEdges, normalizedCaps, edges

def initTrimapFromBBox(img, bbox):
        trimap = np.ones(img.shape[:2])
        trimap *= -1

        for x in range(bbox[0], bbox[2]):
                for y in range(bbox[1], bbox[3]):
                        trimap[y, x] = 0

        return trimap

def floodFill(mask, pt, value):
        for dim in range(len(mask.shape)):
                if pt[dim] < 0 or pt[dim] >= mask.shape[dim]:
                        return

        if mask[pt] == 1:
                mask[pt] = value
        else:
                return

        floodFill(mask, (pt[0] + 1, pt[1]), value)
        floodFill(mask, (pt[0] - 1, pt[1]), value)
        floodFill(mask, (pt[0], pt[1] + 1), value)
        floodFill(mask, (pt[0], pt[1] - 1), value)

def removeAllButLargestComponent(mask):
        mask = np.asarray(mask, dtype="int8")

        index = 2 #These indices will be used to label the components

        image = cv2.cv.fromarray(mask)
        for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                        if mask[y, x] == 1:
                                #cv2.cv.FloodFill(image=image, seed_point=(y, x), new_val=index) #Fill the component with the current index
                                #cv2.floodFill(image=mask, mask=np.zeros((mask.shape[0]+3, mask.shape[1]+2)), seedPoint=(x, y), newVal=index)
                                floodFill(mask, (y, x), index)
                                index += 1

        counts = np.bincount(mask.flatten())
        bestIndex = np.argmax(counts[1:]) + 1

        mask[mask != bestIndex] = 0 #Eliminate all but the largest component

        return mask

def fitGMM(obs):
        numComponents = min(settings.numComponents, len(obs))

        gmm = sklearn.mixture.VBGMM(n_components=numComponents, covariance_type=settings.covType, random_state=None, thresh=0.001, min_covar=0.001, n_iter=10, params='wmc', init_params='wmc')
        gmm.fit(obs)

        return gmm

def calcMaskUsingMyGrabCut(img, bbox, filename):
        ftype = ".jpg"

        trimap = initTrimapFromBBox(img, bbox)
        pmask = np.zeros(trimap.shape)

        mask = np.copy(trimap)
        mask += 1

        binaryEdges, binaryCapacities, edges = initBinaryEdges(img)

        iteration = 0
        while differenceBetweenTwoMasks(pmask, mask) > .01 or iteration < 3:
                print "Difference between masks", differenceBetweenTwoMasks(pmask, mask)
                print("Beginning " + filename + " on iteration " + str(iteration))

                pmask = np.copy(mask)

                fgObs = img[mask != 0]
                bgObs = img[mask == 0]



                print("Making GMMs...")
                fgProb = np.zeros(mask.shape)
                if len(fgObs) >= 0:
                        for _ in range(settings.numGMMs):
                                fgProb += fitGMM(fgObs).score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)

                bgProb = np.zeros(mask.shape)
                if len(bgObs) > 0:
                        for _ in range(settings.numGMMs):
                                bgProb += fitGMM(bgObs).score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)

                #Visualize the image mapped to best components of one gaussian mixture model or the other
                fgProb = normalizeArr(fgProb)
                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "fgProb" + ftype, fgProb)
                bgProb = normalizeArr(bgProb)
                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "bgProb" + ftype, bgProb)

                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "edges" + ftype, edges)



                binaryEdgesArr = np.zeros(mask.shape)
                for edge, cap in zip(binaryEdges, binaryCapacities):
                        binaryEdgesArr[edge[0]] += cap
                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "y" + ftype, binaryEdgesArr)

                #Create edges to source and sink
                unaryTerm = fgProb - bgProb
                unaryTerm = gaussianBlur(unaryTerm, sigma=3) #Blur the unary term to remove high frequency noise
                unaryTerm = normalizeArr(unaryTerm, shift=True)
                unaryTerm += settings.bias
                """
		print("Binary edges range from " + str(min(binaryCapacities)) + " to " + str(max(binaryCapacities)) + "\nwith median of " + str(np.median(binaryCapacities)) + " and average of " + str(np.average(binaryCapacities)))
		print("Unary edges range from " + str(np.min(unaryTerm)) + " to " + str(np.max(unaryTerm)) + "\nwith median of " + str(np.median(unaryTerm)) + " and average of " + str(np.average(unaryTerm)))
		"""
                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "unaryTerm" + ftype, unaryTerm)

                REALLY_BIG_CAPACITY = 10000
                unaryTerm[trimap == -1] = -REALLY_BIG_CAPACITY
                unaryTerm[trimap == 1] = REALLY_BIG_CAPACITY

                assert np.min(unaryTerm) < 0
                assert np.max(unaryTerm) > 0

                unaryEdges = []
                unaryCapacities = []
                for x in range(mask.shape[1]):
                        for y in range(mask.shape[0]):
                                uterm = unaryTerm[y, x]

                                if uterm > 0:
                                        unaryEdges.append(("source", (y,x)))
                                        unaryCapacities.append(uterm)
                                elif uterm < 0:
                                        unaryEdges.append(((y,x), "sink"))
                                        unaryCapacities.append(-uterm)

                vertexList = [(y, x) for x in range(mask.shape[1]) for y in range(mask.shape[0])]

                assert max(binaryCapacities + unaryCapacities) <= REALLY_BIG_CAPACITY
                assert min(binaryCapacities + unaryCapacities) >= 0

                cuts = minCuts(vertexList, binaryEdges + unaryEdges, binaryCapacities + unaryCapacities)

                if len(cuts) == 0:
                        print("No cuts found! Trying again anyway....")
                        print("")
                        mask[...] = 0
                        mask[unaryTerm > .00001] = 1
                else:
                        cut = cuts[0]

                        print(cut)

                        mask[...] = 0 #zero out the mask
                        for vertexIndex in cut[0]:
                                if vertexIndex < len(vertexList):
                                        pt = vertexList[vertexIndex]
                                        mask[pt] = 1

                directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "z" + ftype, mask)
                print("")

                iteration += 1

        mask = removeAllButLargestComponent(mask)
        return mask

def differenceBetweenTwoMasks(m1, m2):
        assert m1.size == m2.size

        return np.sum(np.abs(m1 - m2))/float(m1.size)

def myGrabCut(img, bbox, filename):
        mask = calcMaskUsingMyGrabCut(img=sp.misc.imresize(img, float(1)/settings.sizeRatio), bbox=[x/settings.sizeRatio for x in bbox], filename=filename)

        scaledUp = sp.misc.imresize(mask, img.shape[:2])

        scaledUp = gaussianBlur(scaledUp, sigma=15)

        scaledUp = threshold(scaledUp, threshold=150)

        return scaledUp

def openCVGrabCut(img, bbox):
        mask = np.zeros(img.shape[:2],dtype='uint8')
        tmp1 = np.zeros((1, 13 * 5))
        tmp2 = np.zeros((1, 13 * 5))

        cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
        return mask

def main():
        directories.clearFolder(directories.test)

        print("Running GrabCut...")

        for img, bbox, filename in ImagesAndBBoxes:
        #for img, bbox, filename in [ImagesAndBBoxes[0], ImagesAndBBoxes[5]]:
                bbox = map(int, bbox)
                bbox = tuple(bbox)

                #mask = openCVGrabCut(img, bbox)
                mask = myGrabCut(img, bbox, filename)

                #result = cv2.Image(mask)
                print("Finished one image.")
                directories.saveArrayAsImage(directories.output + filename + ".bmp", mask)

        print("GrabCut is finished :D")

if __name__ == "__main__":
        main()