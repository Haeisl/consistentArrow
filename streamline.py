from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
from vtk.vtkCommonCore import vtkDoubleArray
from pyprtl.models.ModelBase import *
import numpy as np
from random import uniform
import math
import vtk


from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain


@smproxy.filter(label="streamline")
@smproperty.input(name="Input")
# Wird nicht für Funktionalität benötigt, aber so könnt ihr den Output in der GUI umbenennen:
@smproperty.xml("""<OutputPort name="Output" index="0" id="port0" />""")
@smproperty.xml("""<OutputPort name="PolyOutput" index="1" id="port1" />""")
class streamline(VTKPythonAlgorithmBase):
    def __init__(self):
        self._start_point = [1., 1., 1.]
        self._variant = 0
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=2)
    
    @smproperty.doublevector(name="StartPoint", default_values=[1., 1., 1.])
    @smdomain.doublerange()
    def SetStartPoint(self, x, y, z):
        self._start_point = [x, y, z]
        self.Modified()
    
    # hi i bims, ender max fals mer :)
    @smproperty.intvector(name="Variant", default_values=0)
    @smdomain.intrange(min=0, max=5)
    def SetVariant(self, i):
        self._variant = i
        self.Modified()
        
    
    def isInBounds(self, bounds, point):
        return not(
            (point[0] < bounds[0] or point[0] > bounds[1]) or
            (point[1] < bounds[2] or point[1] > bounds[3]) or
            (point[2] < bounds[4] or point[2] > bounds[5])
            )
    
    def createRandomStartPoints(self, input0, numberOfPts):
        bounds = input0.GetBounds()
        randPts = []
        for i in range(numberOfPts):
            randPts.append([uniform(bounds[0], bounds[1]),
                           uniform(bounds[2], bounds[3]),
                           uniform(bounds[4], bounds[5])])
        return randPts
    
    def getTensorFromPoint(self, input0, point):
        # get tensor/tuple9 at given point
        dimensions = input0.GetDimensions()
        bounds = input0.GetBounds()
        spacing = input0.GetSpacing()
        origin = input0.GetOrigin()
        
        if not self.isInBounds(bounds, point):
            print("Point out of bounds")
            return "no"
        
        x = round((point[0] - origin[0])/spacing[0])
        y = round((point[1] - origin[1])/spacing[1])
        z = round((point[2] - origin[2])/spacing[2])
        idx = x + y * dimensions[0] + z * dimensions[0] * dimensions[1]
        return input0.GetPointData().GetArray(0).GetTuple9(idx)
    
    def calcEigenvectors(self, input0, point):
            tuple = self.getTensorFromPoint(input0, point)
            if tuple == "no":
                print("bad")
                return [[1,1,1],[1,1,1],[1,1,1]]
            
            comps = len(tuple)
            npmatrix = np.array([[tuple[i] for i in range(0, comps, 3)],[tuple[i] for i in range(1, comps, 3)],[tuple[i] for i in range(2, comps, 3)]])
            # eigenvectors -> matrix where column eigenvectors[:,i] is normalized eigenvector corresponding to eigenvalues[i]
            eigenValues,eigenVectors = np.linalg.eig(npmatrix)
            
            idx, =  np.where(eigenValues == max(eigenValues))
            
            ind = eigenValues.argsort()[::-1]
            #sortedEigenValues = eigenValues[ind]
            sortedEigenVectors = eigenVectors[:,ind]

            return sortedEigenVectors
    
    def getClosestPointTo(self, input0, refPoint):
        # TODO: trilinear interpolation
        # when there is no datapoint/tensor at the reference point
        # return interpolated tensor for reference point
        bounds = input0.GetBounds()
        spacing = input0.GetSpacing()
        # same as the following
        # if refPoint[0] < bounds[0]: refPoint[0] = bounds[0]
        # if refPoint[0] > bounds[1]: refPoint[0] = bounds[1]
        # if refPoint[1] < bounds[2]: refPoint[1] = bounds[2]
        # if refPoint[1] > bounds[3]: refPoint[1] = bounds[3]
        # if refPoint[2] < bounds[4]: refPoint[2] = bounds[4]
        # if refPoint[2] > bounds[5]: refPoint[2] = bounds[5]
        # tests whether a coordinate is outside the bounds and 'clips' it, such that the coordinate at the corr. boundary is taken
        for i in range(6):
            coord = round(np.floor(i/2.))
            print(coord)
            if i % 2 == 0:
                if refPoint[coord] < bounds[i]: refPoint[coord] = bounds[i]
            else:
                if refPoint[coord] > bounds[i]: refPoint[coord] = bounds[i]
        
        rx = np.around(refPoint[0], decimals=1)
        ry = np.around(refPoint[1], decimals=1)
        rz = np.around(refPoint[2], decimals=1)
    
        return [rx, ry, rz]

        
    def FillOutputPortInformation(self, port, info):
        if port == 0:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkImageData')
        elif port == 1:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkPolyData')
        return 1
    
    def buildLine(self, linePointArray, pLine, idx, c, points):
            for point in points:
                linePointArray.InsertNextPoint(point)
                idx += 1
                pLine.GetPointIds().InsertNextId(idx)

                if self._variant != 0:
                    linePointArray.InsertNextPoint(c)
                    idx += 1
                    pLine.GetPointIds().InsertNextId(idx)
                
            return idx
    
    def RequestData(self, request, inInfo, outInfo):
        # get the first input
        input0 = vtkImageData.GetData(inInfo[0])
        
        # get the outputs
        output = vtkImageData.GetData(outInfo, 0)
        output.ShallowCopy(input0)
        
        polyOutput = vtkPolyData.GetData(outInfo, 1)

        bounds = input0.GetBounds()
        
        linePoints = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        polyline = vtk.vtkPolyLine()
        
        quadPoints = vtk.vtkPoints()
        quads = vtk.vtkCellArray()
        quad = vtk.vtkQuad()
        #colors = vtk.vtkNamedColors()
    
        ptsArr = self.createRandomStartPoints(input0, 20)
        
        pointsToCompute = [self._start_point]
        #for pts in ptsArr:
        #    pointsToCompute.append(pts)
        #pointsToCompute.append([-1.,1.,1.])
        #print(pts)
        
        total_points = 0
        #total_quad_points = -1
        cur = self._start_point
        linePoints.InsertNextPoint(cur)
        #polyline.GetPointIds().InsertNextId(total_points)
        
        scaling = 1/10

        outOfBounds = False
        while not outOfBounds:
            eigenVectors = self.calcEigenvectors(input0, cur)
            firstEigenVector = eigenVectors[:,0]
            secondEigenVector = eigenVectors[:,1]
            thirdEigenVector = eigenVectors[:,2]
            
            #next = self.getClosestPointTo(input0, np.add(cur, largeEigenVector))
            next = np.add(cur, firstEigenVector*scaling)
            
            if self.isInBounds(bounds, next):
                # start at next point of previous iteration
                # line from one -> two -> start -> three -> four -> start -> next -> ...
                halfSecondEV = 0.5*secondEigenVector
                halfThirdEV = 0.5*thirdEigenVector

                pointOne = np.add(cur, -halfSecondEV*scaling)
                pointTwo = np.add(cur, halfSecondEV*scaling)
                pointThree = np.add(cur, -halfThirdEV*scaling)
                pointFour = np.add(cur, halfThirdEV*scaling)

                # only first eigenvector
                if self._variant == 0:
                    points = [next]
                # first eigenvector, second eigenvector in natural orientation
                elif self._variant == 1:
                    points = [np.add(pointTwo, halfSecondEV*scaling)]
                # first eigenvector, second eigenvector moved s.t. center is at the point
                elif self._variant == 2:
                    points = [pointOne, pointTwo]
                # first eigenvector, second eigenvector moved s.t. center is at the point, third eigenvector in natrual orientation
                elif self._variant == 3:
                    points = [pointOne, pointTwo, np.add(pointFour, halfThirdEV*scaling)]
                # first, second and third eigenvector moved s.t. center ist at the point
                elif self._variant == 4:
                    points = [pointOne, pointTwo, pointThree, pointFour]
                # first eigenvector, second and third eigenvector in their natural orientation
                elif self._variant == 5:
                    points = [np.add(pointTwo, halfSecondEV*scaling), np.add(pointFour, halfThirdEV*scaling)]
                else:
                    print("no such variant")
                    return 1
                
                total_points = self.buildLine(linePoints, polyline, total_points, cur, points)

                linePoints.InsertNextPoint(next)
                total_points += 1
                polyline.GetPointIds().InsertNextId(total_points)
                cur = next
                
                #linePoints.InsertNextPoint(next)
                #total_points += 1
                #polyline.GetPointIds().InsertNextId(total_points)

                #cur = next
            else:
                outOfBounds = True

        lines.InsertNextCell(polyline)
        #quads.InsertNextCell(quad)

        #polyOutput.SetPoints(quadPoints)
        polyOutput.SetPoints(linePoints)
        polyOutput.SetLines(lines)

        #polyOutput.SetPolys(quads)
        
        #print(linePoints.GetNumberOfPoints())
        
        return 1