from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
from vtk.vtkCommonCore import vtkDoubleArray
from pyprtl.models.ModelBase import *
import numpy as np
import math
import vtk
import time
#import multiprocessing


from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(label="consistentArrow")
@smproperty.input(name="Input")
@smproperty.xml("""<OutputPort name="PolyOutput" index="0" id="port0" />""")
class consistentArrow(VTKPythonAlgorithmBase):
    def __init__(self):
        self._mode = "rk4"
        self._center = [1., 1.]
        self._thickness = 1
        self._length = 7
        self._steps = 2
        self._cols = 1
        self._rows = 1
        '''
        self._granularity = 0.05
        self._tipLen = 5
        self._ratioL = 0.5
        self._ratioR = 0.5
        '''
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

    @smproperty.stringvector(name="StringInfo", information_only="1")
    def GetStrings(self):
        return ["euler", "rk4"]

    @smproperty.stringvector(name="Variant", number_of_elements="1", default="rk4")
    @smdomain.xml(\
        """<StringListDomain name="list">
                <RequiredProperties>
                    <Property name="StringInfo" function="StringInfo"/>
                </RequiredProperties>
            </StringListDomain>
        """)
    def SetString(self, value):
        self._mode = value
        self.Modified()

    @smproperty.doublevector(name="Center Point", default_values=[1., 1.])
    @smdomain.doublerange()
    def SetStartPoint(self, x, y):
        self._center = [x, y]
        self.Modified()

    @smproperty.intvector(name="Steps To Side", number_of_elements=1, default_values=1)
    @smdomain.intrange(min=0, max=5)
    def SetThickness(self, d):
        self._thickness = d
        self.Modified()

    @smproperty.intvector(name="Steps From Center", number_of_elements=1, default_values=7)
    @smdomain.intrange(min=1, max=10)
    def SetLength(self, l):
        self._length = l
        self.Modified()

    @smproperty.intvector(name="Grid Points Horizontal", number_of_elements=1, default_values=1)
    @smdomain.intrange(min=1,max=10)
    def SetCols(self, c):
        self._cols = c
        self.Modified()

    @smproperty.intvector(name="Grid Points Vertical", number_of_elements=1, default_values=1)
    @smdomain.intrange(min=1,max=10)
    def SetRows(self, r):
        self._rows = r
        self.Modified()

    @smproperty.intvector(name="Steps per One", number_of_elements=1, default_values=2)
    @smdomain.intrange(min=1, max=10)
    def SetGrain(self, d):
        self._steps = d
        self.Modified()

    def FillOutputPortInformation(self, port, info):
        info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkPolyData')
        return 1

    def generateGridPoints(self, bounds):

        breadth = abs(bounds[0]) + abs(bounds[1])
        height = abs(bounds[2]) + abs(bounds[3])

        distX = breadth / (self._cols + 1)
        distY = height / (self._rows + 1)

        centerBreadth = breadth / 2
        centerHeight = height / 2

        yComponents = []
        xComponents = []

        if self._rows % 2 == 1:
            yComponents.append(centerHeight)
            for i in range(int((self._rows - 1) / 2)):
                yComponents.append(centerHeight - (i + 1) * distY)
                yComponents.append(centerHeight + (i + 1) * distY)
        else:
            evenHeightLower = centerHeight - 0.5 * distY
            evenHeightUpper = centerHeight + 0.5 * distY
            yComponents.append(evenHeightLower)
            yComponents.append(evenHeightUpper)
            for i in range(int(self._rows / 2) - 1):
                yComponents.append(evenHeightLower - (i + 1) * distY)
                yComponents.append(evenHeightUpper + (i + 1) * distY)

        yComponents.sort()
        print(yComponents)

        if self._cols % 2 == 1:
            xComponents.append(centerBreadth)
            for i in range(int((self._cols - 1) / 2)):
                xComponents.append(centerBreadth - (i + 1) * distX)
                xComponents.append(centerBreadth + (i + 1) * distX)
        else:
            evenBreadthLower = centerBreadth - 0.5 * distX
            evenBreadthUpper = centerBreadth + 0.5 * distX
            xComponents.append(evenBreadthLower)
            xComponents.append(evenBreadthUpper)
            for i in range(int(self._cols / 2) - 1):
                xComponents.append(evenBreadthLower - (i + 1) * distX)
                xComponents.append(evenBreadthUpper + (i + 1) * distX)

        xComponents.sort()
        print(xComponents)

        grid = []
        for i in range(len(xComponents)):
            for j in range(len(yComponents)):
                grid.append([xComponents[i], yComponents[j]])

        return grid

    def findIndex(self, dims, bounds, point):
        # dims = input0.GetDimensions()
        # bounds = input0.GetBounds()

        xdir = np.linspace(bounds[0], bounds[1], dims[0])
        ydir = np.linspace(bounds[2], bounds[3], dims[1])

        index_px = np.searchsorted(xdir, point[0])
        index_py = np.searchsorted(ydir, point[1])

        return [index_px-1, index_py-1]

    def bilinear_interpolation(self, image_data, position):
        """
        Bilinear interpolation for obtaining the vector at a given position in the vector field.

        Parameters:
            image_data: vtk.vtkImageData
                vtkImageData object containing the vector field data.
            position: numpy.ndarray
                Position at which to interpolate the vector. Shape: (2,)
                The position should be in the image data coordinate system

        Returns:
            numpy.ndarray
                Vector interpolated at the given position
        """
        # extract image dimensions
        dims = image_data.GetDimensions()
        i, j = position

        # extract grid coordinates
        i_floor, j_floor = int(np.floor(i)), int(np.floor(j))
        i_ceil, j_ceil  = int(np.ceil(i)), int(np.ceil(j))

        # bilinear interpolation weights
        di = i - i_floor
        dj = j - j_floor

        # interpolate vectors
        vector_field = image_data.GetPointData().GetArray(0)
        vector_floor_floor = np.array(vector_field.GetTuple(np.ravel_multi_index((i_floor, j_floor, 0), dims)))
        vector_floor_ceil = np.array(vector_field.GetTuple(np.ravel_multi_index((i_floor, j_ceil, 0), dims)))
        vector_ceil_floor = np.array(vector_field.GetTuple(np.ravel_multi_index((i_ceil, j_floor, 0), dims)))
        vector_ceil_ceil = np.array(vector_field.GetTuple(np.ravel_multi_index((i_ceil, j_ceil, 0), dims)))

        interpolated_vector = (1 - di) * ((1 - dj) * vector_floor_floor + dj * vector_ceil_floor) + \
                                di * ((1 - dj) * vector_floor_ceil + dj * vector_ceil_ceil)

        return interpolated_vector


    # def GetInterpVector(self, input0, point, normalized=False):
    #     # utilizing bilinear interpolation
    #     dimensions = input0.GetDimensions()
    #     bounds = input0.GetBounds()
    #     spacing = input0.GetSpacing()
    #     origin = input0.GetOrigin()

    #     origin_0 = origin[0]
    #     origin_1 = origin[1]
    #     spacing_0 = spacing[0]
    #     spacing_1 = spacing[1]
    #     dimensions_0 = dimensions[0]

    #     if point[0] < bounds[0]:
    #         point[0] = bounds[0]

    #     elif point[0] > bounds[1]:
    #         point[0] = bounds[1]

    #     if point[1] < bounds[2]:
    #         point[1] = bounds[2]

    #     elif point[1] > bounds[3]:
    #         point[1] = bounds[3]

    #     indices = self.findIndex(dimensions, bounds, point)

    #     px = point[0]
    #     py = point[1]

    #     x1 = indices[0] * spacing[0] - abs(bounds[0])
    #     x2 = (indices[0] + 1) * spacing[0] - abs(bounds[0])
    #     y1 = indices[1] * spacing[1] - abs(bounds[1])
    #     y2 = (indices[1] + 1) * spacing[1] - abs(bounds[1])

    #     q11 = input0.GetPointData().GetArray(0).GetTuple(
    #         int(
    #             (x1 - origin_0)/spacing_0 +
    #             (y1 - origin_1)/spacing_1 * dimensions_0
    #         )
    #     )
    #     q12 = input0.GetPointData().GetArray(0).GetTuple(
    #         int(
    #             (x1 - origin_0)/spacing_0 +
    #             (y2 - origin_1)/spacing_1 * dimensions_0
    #         )
    #     )
    #     q21 = input0.GetPointData().GetArray(0).GetTuple(
    #         int(
    #             (x2 - origin_0)/spacing_0 +
    #             (y1 - origin_1)/spacing_1 * dimensions_0
    #         )
    #     )
    #     q22 = input0.GetPointData().GetArray(0).GetTuple(
    #         int(
    #             (x2 - origin_0)/spacing_0 +
    #             (y2 - origin_1)/spacing_1 * dimensions_0
    #         )
    #     )

    #     interpolated = (
    #         np.divide(
    #             (np.multiply(q11, (x2 - px) * (y2 - py)) +
    #             np.multiply(q21, (px - x1) * (y2 - py)) +
    #             np.multiply(q12, (x2 - px) * (py - y1)) +
    #             np.multiply(q22, (px - x1) * (py - y1))), ((x2 - x1) * (y2 - y1) + 0.0))
    #     )
    #     interpolatedVec = [interpolated[0], interpolated[1], 0.]

    #     if normalized:
    #         return interpolatedVec / np.linalg.norm(interpolatedVec)

    #     return interpolatedVec

    # def ClipAtBounds(self, point, input0, previous=[0.,0.]):
    #     bounds = input0.GetBounds()
    #     if point[0] < bounds[0]:
    #         point[0] = bounds[0]

    #     elif point[0] > bounds[1]:
    #         point[0] = bounds[1]

    #     if point[1] < bounds[2]:
    #         point[1] = bounds[2]

    #     elif point[1] > bounds[3]:
    #         point[1] = bounds[3]

    #     if point[2] != 0.:
    #         point[2] = 0.

    def rk4_integrate(self, image_data):


        def rotate_vector(vector, direction):
            if direction == "clockwise":
                return np.array([vector[1], -vector[0]])
            elif direction == "counterclockwise":
                return np.array([-vector[1], vector[0]])
            else:
                raise ValueError("Invalid rotation direction.")

        def integrate_step(point, h, direction):
            k1 = np.array(self.bilinear_interpolation(image_data, point))
            k2 = np.array(self.bilinear_interpolation(image_data, point + h / 2 * k1))
            k3 = np.array(self.bilinear_interpolation(image_data, point + h / 2 * k2))
            k4 = np.array(self.bilinear_interpolation(image_data, point + h * k3))

            return point + h / 6 * (k1 + 2 * k2 + 2 * k3)

    def rk4(
        self,
        input0: vtkImageData,
        pointArr: list,
        loopLen: int,
        cur: list,
        stepsize: float,
        goingForward:bool,
        goingLeft:bool,
        indexFactor: int,
        indexIncr: int,
        scaling: float,
        orthogonal: bool,
    ) -> list:
        if orthogonal:
            directionForward = pow(-1, not goingForward)
            directionLeft = pow(-1, goingLeft)
            i = 0
            t = 0
            while t < loopLen - 1e-4:
                k_1 = self.GetInterpVector(input0, cur)
                k_1 = np.multiply([directionLeft * k_1[1], (-1 * directionLeft) * k_1[0], 0.], scaling)

                k_2 = self.GetInterpVector(input0, point=[cur[0] + (k_1[0]/2.), cur[1] + (k_1[1]/2.), 0.])
                k_2 = np.multiply([directionLeft * k_2[1], (-1 * directionLeft) * k_2[0], 0.], scaling)

                k_3 = self.GetInterpVector(input0, point=[cur[0] + (k_2[0]/2.), cur[1] + (k_2[1]/2.), 0.])
                k_3 = np.multiply([directionLeft * k_3[1], (-1 * directionLeft) * k_3[0], 0.], scaling)

                k_4 = self.GetInterpVector(input0, point=[cur[0] + k_3[0], cur[1] + k_3[1], 0.])
                k_4 = np.multiply([directionLeft * k_4[1], (-1 * directionLeft) * k_4[0], 0.], scaling)

                next = np.add(cur, directionForward * (stepsize/6.)*(np.add(k_1,
                                                                            np.add(2*k_2,
                                                                                np.add(2*k_3, k_4)
                                                                                )
                                                                            )
                                                                    )
                            )
                self.ClipAtBounds(next, input0, cur)
                pointArr[indexIncr + indexFactor * (i + 1)] = next
                cur = next
                i += 1
                t += stepsize
        else:
            directionForward = pow(-1, not goingForward)
            i = 0
            t = 0
            while t < loopLen - 1e-4:
                k_1 = np.multiply(self.GetInterpVector(input0, cur), scaling)

                k_2 = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k_1[0]/2.), cur[1] + (k_1[1]/2.), 0.]), scaling)

                k_3 = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k_2[0]/2.), cur[1] + (k_2[1]/2.), 0.]), scaling)

                k_4 = np.multiply(self.GetInterpVector(input0, point=[cur[0] + k_3[0], cur[1] + k_3[1], 0.]), scaling)

                next = np.add(cur, directionForward * (stepsize/6.)*(np.add(k_1,
                                                                            np.add(2*k_2,
                                                                                np.add(2*k_3, k_4)
                                                                                )
                                                                            )
                                                                    )
                            )
                self.ClipAtBounds(next, input0, cur)
                pointArr[indexIncr + indexFactor * (i + 1)] = next
                cur = next
                i += 1
                t += stepsize

        return pointArr


    def euler(
        self,
        input0: vtkImageData,
        pointArr: list,
        loopLen: int,
        cur: list,
        stepsize: float,
        goingForward: bool,
        goingLeft: bool,
        indexFactor: int,
        indexIncr: int,
        scaling: float,
        orthogonal: bool,
    ) -> list:
        if orthogonal:
            directionForward = pow(-1, not goingForward)
            directionLeft = pow(-1, goingLeft)
            i = 0
            t = 0
            while t < loopLen - 1e-4:
                vec = self.GetInterpVector(input0, cur)
                vec = [directionLeft * vec[1], (-1 * directionLeft) * vec[0], 0.]
                next = np.add(cur, directionForward * np.multiply(np.multiply(vec, stepsize), scaling))
                self.ClipAtBounds(next, input0, cur)
                pointArr[indexIncr + indexFactor * (i + 1)] = next
                cur = next
                i += 1
                t += stepsize
        else:
            directionForward = pow(-1, not goingForward)
            i = 0
            t = 0
            while t < loopLen - 1e-4:
                vec = self.GetInterpVector(input0, cur)
                next = np.add(cur, directionForward * np.multiply(np.multiply(vec, stepsize), scaling))
                self.ClipAtBounds(next, input0, cur)
                pointArr[indexIncr + indexFactor * (i + 1)] = next
                cur = next
                i += 1
                t += stepsize

        return pointArr

    def setNumberOfIdsAndLinkToPoints(
        self,
        centerLineLength,
        centerPolyline,
        bottomArcLength,
        bottomEdge,
        leftLineLength,
        leftEdge,
        rightLineLength,
        rightEdge,
        leftArrowBaseLength,
        arrowBaseL,
        rightArrowBaseLength,
        arrowBaseR
    ):
        centerPolyline.GetPointIds().SetNumberOfIds(centerLineLength)
        bottomEdge.GetPointIds().SetNumberOfIds(bottomArcLength)
        leftEdge.GetPointIds().SetNumberOfIds(leftLineLength)
        rightEdge.GetPointIds().SetNumberOfIds(rightLineLength)
        arrowBaseL.GetPointIds().SetNumberOfIds(leftArrowBaseLength)
        arrowBaseR.GetPointIds().SetNumberOfIds(rightArrowBaseLength)

        for i in range(centerLineLength):
            centerPolyline.GetPointIds().SetId(i, i)

        for i in range(bottomArcLength):
            bottomEdge.GetPointIds().SetId(i, centerLineLength + i)

        for i in range(leftLineLength):
            leftEdge.GetPointIds().SetId(i, centerLineLength + bottomArcLength + i)

        for i in range(rightLineLength):
            rightEdge.GetPointIds().SetId(i, centerLineLength + bottomArcLength + leftLineLength + i)

        for i in range(leftArrowBaseLength):
            arrowBaseL.GetPointIds().SetId(i, centerLineLength + bottomArcLength + leftLineLength + rightLineLength + i)

        for i in range(rightArrowBaseLength):
            arrowBaseR.GetPointIds().SetId(i, centerLineLength + bottomArcLength + leftLineLength + rightLineLength + leftArrowBaseLength + i)


    def RequestData(self, request, inInfo, outInfo):
        # timing
        startTime = time.time()

        # get the first input
        input0 = vtkImageData.GetData(inInfo[0])

        # get the output
        polyOutput = vtkPolyData.GetData(outInfo)

        bounds = input0.GetBounds()

        linePoints = vtk.vtkPoints()

        bottomEdge = vtk.vtkPolyLine()
        centerPolyline = vtk.vtkPolyLine()
        leftEdge = vtk.vtkPolyLine()
        rightEdge = vtk.vtkPolyLine()
        arrowBaseL = vtk.vtkPolyLine()
        arrowBaseR = vtk.vtkPolyLine()
        arrowTipL = vtk.vtkPolyLine()
        arrowTipR = vtk.vtkPolyLine()

        thickness = int(self._thickness)
        startCenter = [self._center[0], self._center[1], 0.]
        scaling = 1/50
        arrowheadStart = int(math.ceil(self._length/2.0)) - 1

        grid = self.generateGridPoints(bounds)

        ############################################################ SETTING UP POINT ARRAYS FOR LINES ############################################################
        # line generation order:
        # center line -> left/right arc -> left/right line -> left/right arrowbase -> left/right arrowhead
        #
        # relevant points/indices:
        # ORIGIN = point in the middle of center line
        # CENTER_START = point at the start of center line
        # LEFT_START = point at the start of left line
        # RIGHT_START = point at the start of right line
        # LEFT_END = point at the end of left line
        # RIGHT_END = point at the end of right line
        # AB_LEFT_END = point at the end of the left arrowbase
        # AB_RIGHT_END = point at the end of the right arrowbase

        steps = self._steps
        stepsize = 1/steps

        ORIGIN = startCenter

        centerLineLength = 2 * (self._length * steps) + 1
        centerLinePoints = [None] * centerLineLength
        for i in [1, 0]:
            # first run: tracing back, second run: forth
            cur = ORIGIN
            directionFactor = pow(-1, i) # -1 for first pass -> back | 1 for second pass -> forth
            if self._mode == "euler":
                centerLinePoints = self.euler(
                    input0=input0,
                    pointArr=centerLinePoints,
                    loopLen=self._length,
                    cur=cur,
                    stepsize=stepsize,
                    goingForward=not bool(i),
                    goingLeft=None,
                    indexFactor=directionFactor,
                    indexIncr=self._length*steps,
                    scaling=scaling,
                    orthogonal=False
                )
            elif self._mode == "rk4":
                centerLinePoints = self.rk4(
                    input0=input0,
                    pointArr=centerLinePoints,
                    loopLen=self._length,
                    cur=cur,
                    stepsize=stepsize,
                    goingForward=not bool(i),
                    goingLeft=None,
                    indexFactor=directionFactor,
                    indexIncr=self._length*steps,
                    scaling=scaling,
                    orthogonal=False
                )
            centerLinePoints[int(self._length*steps)] = ORIGIN

        CENTER_START = centerLinePoints[0]
        CENTER_END = centerLinePoints[len(centerLinePoints) - 1]


        bottomArcLength = 2 * (thickness*steps) + 1
        bottomArcPoints = [None] * bottomArcLength
        for i in [1, 0]:
            # first part: left, second run: right
            cur = CENTER_START
            directionFactor = pow(-1, i) # -1 for first pass -> back | 1 for second pass -> forth
            if self._mode == "euler":
                bottomArcPoints = self.euler(
                    input0=input0,
                    pointArr=bottomArcPoints,
                    loopLen=thickness,
                    cur=cur,
                    stepsize=stepsize,
                    goingForward=True,
                    goingLeft=bool(i),
                    indexFactor=directionFactor,
                    indexIncr=thickness*steps,
                    scaling=scaling,
                    orthogonal=True
                )
            elif self._mode == "rk4":
                bottomArcPoints = self.rk4(
                    input0=input0,
                    pointArr=bottomArcPoints,
                    loopLen=thickness,
                    cur=cur,
                    stepsize=stepsize,
                    goingForward=True,
                    goingLeft=bool(i),
                    indexFactor=directionFactor,
                    indexIncr=thickness*steps,
                    scaling=scaling,
                    orthogonal=True
                )
            bottomArcPoints[thickness*steps] = CENTER_START


        LEFT_START = bottomArcPoints[0]
        RIGHT_START = bottomArcPoints[len(bottomArcPoints) - 1]


        # arrowheadStart is after how many points after the origin the arrowhead will start
        leftLineLength = steps * (self._length + arrowheadStart) + 1
        leftLinePoints = [None] * leftLineLength
        cur = LEFT_START
        leftLinePoints[0] = LEFT_START
        if self._mode == "euler":
            leftLinePoints = self.euler(
                input0=input0,
                pointArr=leftLinePoints,
                loopLen=self._length + arrowheadStart,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=None,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=False
            )
        elif self._mode == "rk4":
            leftLinePoints = self.rk4(
                input0=input0,
                pointArr=leftLinePoints,
                loopLen=self._length + arrowheadStart,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=None,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=False
            )

        LEFT_END = leftLinePoints[len(leftLinePoints) - 1]


        rightLineLength = steps*(self._length + arrowheadStart) + 1
        rightLinePoints = [None] * rightLineLength
        cur = RIGHT_START
        rightLinePoints[0] = RIGHT_START
        if self._mode == "euler":
            rightLinePoints = self.euler(
                input0=input0,
                pointArr=rightLinePoints,
                loopLen=self._length + arrowheadStart,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=None,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=False
            )
        elif self._mode == "rk4":
            rightLinePoints = self.rk4(
                input0=input0,
                pointArr=rightLinePoints,
                loopLen=self._length + arrowheadStart,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=None,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=False
            )

        RIGHT_END = rightLinePoints[len(rightLinePoints) - 1]


        leftArrowBaseLength = thickness*steps + 1
        leftArrowBasePoints = [None] * leftArrowBaseLength
        cur = LEFT_END
        leftArrowBasePoints[0] = LEFT_END
        if self._mode == "euler":
            leftArrowBasePoints = self.euler(
                input0=input0,
                pointArr=leftArrowBasePoints,
                loopLen=thickness,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=True,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=True
            )
        elif self._mode == "rk4":
            leftArrowBasePoints = self.rk4(
                input0=input0,
                pointArr=leftArrowBasePoints,
                loopLen=thickness,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=True,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=True
            )

        AB_LEFT_END = leftArrowBasePoints[len(leftArrowBasePoints) - 1]


        rightArrowBaseLength = thickness*steps + 1
        rightArrowBasePoints = [None] * rightArrowBaseLength
        cur = RIGHT_END
        rightArrowBasePoints[0] = RIGHT_END
        if self._mode == "euler":
            rightArrowBasePoints = self.euler(
                input0=input0,
                pointArr=rightArrowBasePoints,
                loopLen=thickness,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=False,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=True
            )
        elif self._mode == "rk4":
            rightArrowBasePoints = self.rk4(
                input0=input0,
                pointArr=rightArrowBasePoints,
                loopLen=thickness,
                cur=cur,
                stepsize=stepsize,
                goingForward=True,
                goingLeft=False,
                indexFactor=1,
                indexIncr=0,
                scaling=scaling,
                orthogonal=True
            )


        AB_RIGHT_END = rightArrowBasePoints[len(rightArrowBasePoints) - 1]


        ############################################################ FILLING VTKPOINTS ARRAY ############################################################
        allPoints = [*centerLinePoints, *bottomArcPoints, *leftLinePoints, *rightLinePoints, *leftArrowBasePoints, *rightArrowBasePoints]
        for point in allPoints:
           linePoints.InsertNextPoint(point)

        ############################################################ SETTING NUMBER OF IDS ############################################################
        ############################################################ SETTING IDS TO POINTS ############################################################

        self.setNumberOfIdsAndLinkToPoints(
            centerLineLength,
            centerPolyline,
            bottomArcLength,
            bottomEdge,
            leftLineLength,
            leftEdge,
            rightLineLength,
            rightEdge,
            leftArrowBaseLength,
            arrowBaseL,
            rightArrowBaseLength,
            arrowBaseR
        )

        ############################################################ ADDING LINES TO CELL ARRAY ############################################################
        lines = vtk.vtkCellArray()

        lines.InsertNextCell(centerPolyline)
        lines.InsertNextCell(bottomEdge)
        lines.InsertNextCell(leftEdge)
        lines.InsertNextCell(rightEdge)

        lines.InsertNextCell(arrowBaseL)
        lines.InsertNextCell(arrowBaseR)

        # add points to the dataset
        polyOutput.SetPoints(linePoints)

        # add lines to the dataset
        polyOutput.SetLines(lines)

        ############################################################ CONSTRUCTING ARROW TIP ############################################################
        def approxEqual(a, b, tolerance=1e-10):
            return abs(a - b) <= tolerance

        def b(factorA):
            return 1 - factorA

        scaling = 1/10
        steps = 20
        stepsize = 1/steps

        threshold = 0.005

        end = [CENTER_END[i] for i in range(len(CENTER_END))]

        satisfied = False
        aFactors = []
        distances = []
        i = 0
        while True:
            if i == 0:
                factorA = 0.
                factorB = b(factorA)
                aFactors.append(factorA)
            elif i == 1:
                factorA = 1.
                factorB = b(factorA)
                aFactors.append(factorA)
            else:
                factorA = (aFactors[0] + aFactors[1])/2.
                factorB = b(factorA)
                aFactors.append(factorA)

            leftArrowHeadPoints = []
            cur = AB_LEFT_END
            leftArrowHeadPoints.append(cur)
            mindist = float('inf')
            while True:
                cur = [cur[i] for i in range(len(cur))]

                k1Parallel = np.multiply(self.GetInterpVector(input0, cur), scaling)
                k1Orthogonal = [k1Parallel[1], -k1Parallel[0], 0.]

                k2Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k1Parallel[0]/2.), cur[1] + (k1Parallel[1]/2.)]), scaling)
                k2Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k1Orthogonal[0]/2.), cur[1] + (k1Orthogonal[1]/2.)]), scaling)
                k2Orthogonal = [k2Orthogonal[1], -k2Orthogonal[0], 0.]

                k3Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k2Parallel[0]/2.), cur[1] + (k2Parallel[1]/2.)]), scaling)
                k3Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k2Orthogonal[0]/2.), cur[1] + (k2Orthogonal[1]/2.)]), scaling)
                k3Orthogonal = [k3Orthogonal[1], -k3Orthogonal[0], 0.]

                k4Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + k3Parallel[0], cur[1] + k3Parallel[1]]), scaling)
                k4Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + k3Orthogonal[0], cur[1] + k3Orthogonal[1]]), scaling)
                k4Orthogonal = [k4Orthogonal[1], -k4Orthogonal[0], 0.]

                stepFac = stepsize/6.
                next = np.add(cur,
                                factorA * stepFac * np.add(k1Parallel, np.add(np.multiply(2,k2Parallel), np.add(np.multiply(2,k3Parallel), k4Parallel))) +
                                factorB * stepFac * np.add(k1Orthogonal, np.add(np.multiply(2,k2Orthogonal), np.add(np.multiply(2,k3Orthogonal), k4Orthogonal)))
                            )


                dist = np.linalg.norm(np.add(end, [-next[i] for i in range(len(next))]))

                if dist <= threshold:
                    leftArrowHeadPoints.append(next)
                    distances.append(dist)
                    satisfied = True
                    break
                if dist <= mindist:
                    leftArrowHeadPoints.append(next)
                    mindist = dist
                else:
                    distances.append(dist)
                    break

                cur = next


            if satisfied:
                break

            if i > 1:
                tooBig = max(distances)
                tooBigInd = distances.index(tooBig)
                distances.remove(tooBig)
                aFactors.remove(aFactors[tooBigInd])

                if approxEqual(distances[0], distances[1]) or approxEqual(aFactors[0], aFactors[1]):
                    break

            i += 1

        #leftArrowHeadPoints.append(CENTER_END)
        leftArrowHeadLength = len(leftArrowHeadPoints)
        AH_LEFT_END = leftArrowHeadPoints[len(leftArrowHeadPoints) - 1]

        # ###########################################################################

        # satisfied = False
        # aFactors = []
        # distances = []
        # i = 0
        # while True:
        #     if i == 0:
        #         factorA = 0.
        #         factorB = b(factorA)
        #         aFactors.append(factorA)
        #     elif i == 1:
        #         factorA = 1.
        #         factorB = b(factorA)
        #         aFactors.append(factorA)
        #     else:
        #         factorA = (aFactors[0] + aFactors[1])/2.
        #         factorB = b(factorA)
        #         aFactors.append(factorA)

        #     rightArrowHeadPoints = []
        #     cur = AB_RIGHT_END
        #     rightArrowHeadPoints.append(cur)
        #     mindist = float('inf')
        #     while True:
        #         cur = [cur[i] for i in range(len(cur))]

        #         k1Parallel = np.multiply(self.GetInterpVector(input0, cur), scaling)
        #         k1Orthogonal = [-k1Parallel[1], k1Parallel[0], 0.]

        #         k2Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k1Parallel[0]/2.), cur[1] + (k1Parallel[1]/2.)]), scaling)
        #         k2Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k1Orthogonal[0]/2.), cur[1] + (k1Orthogonal[1]/2.)]), scaling)
        #         k2Orthogonal = [-k2Orthogonal[1], k2Orthogonal[0], 0.]

        #         k3Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k2Parallel[0]/2.), cur[1] + (k2Parallel[1]/2.)]), scaling)
        #         k3Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + (k2Orthogonal[0]/2.), cur[1] + (k2Orthogonal[1]/2.)]), scaling)
        #         k3Orthogonal = [-k3Orthogonal[1], k3Orthogonal[0], 0.]

        #         k4Parallel = np.multiply(self.GetInterpVector(input0, point=[cur[0] + k3Parallel[0], cur[1] + k3Parallel[1]]), scaling)
        #         k4Orthogonal = np.multiply(self.GetInterpVector(input0, point=[cur[0] + k3Orthogonal[0], cur[1] + k3Orthogonal[1]]), scaling)
        #         k4Orthogonal = [-k4Orthogonal[1], k4Orthogonal[0], 0.]

        #         stepFac = stepsize/6.
        #         next = np.add(cur,
        #                         factorA * stepFac * np.add(k1Parallel, np.add(np.multiply(2,k2Parallel), np.add(np.multiply(2,k3Parallel), k4Parallel))) +
        #                         factorB * stepFac * np.add(k1Orthogonal, np.add(np.multiply(2,k2Orthogonal), np.add(np.multiply(2,k3Orthogonal), k4Orthogonal)))
        #                     )

        #         dist = np.linalg.norm(np.add(end, [-next[i] for i in range(len(next))]))

        #         if dist <= threshold:
        #             rightArrowHeadPoints.append(next)
        #             distances.append(dist)
        #             satisfied = True
        #             break

        #         if dist <= mindist:
        #             rightArrowHeadPoints.append(next)
        #             mindist = dist
        #         else:
        #             distances.append(dist)
        #             break

        #         cur = next

        #     if satisfied:
        #         break

        #     if i > 1:
        #         tooBig = max(distances)
        #         tooBigInd = distances.index(tooBig)
        #         distances.remove(tooBig)
        #         aFactors.remove(aFactors[tooBigInd])

        #         if approxEqual(distances[0], distances[1]) or approxEqual(aFactors[0], aFactors[1]):
        #             break

        #     i += 1

        # #rightArrowHeadPoints.append(CENTER_END)
        # rightArrowHeadLength = len(rightArrowHeadPoints)
        # AH_RIGHT_END = leftArrowHeadPoints[len(leftArrowHeadPoints) - 1]


        arrowHeadPoints = [*leftArrowHeadPoints, *rightArrowBasePoints]
        arrowHeadPoints = leftArrowHeadPoints

        for point in arrowHeadPoints:
            linePoints.InsertNextPoint(point)

        arrowTipL.GetPointIds().SetNumberOfIds(leftArrowHeadLength)
        #arrowTipR.GetPointIds().SetNumberOfIds(rightArrowHeadLength)

        for i in range(leftArrowHeadLength):
            arrowTipL.GetPointIds().SetId(i, centerLineLength + bottomArcLength + leftLineLength + rightLineLength + leftArrowBaseLength + rightArrowBaseLength + i)

        #for i in range(rightArrowHeadLength):
        #    arrowTipR.GetPointIds().SetId(i, centerLineLength + bottomArcLength + leftLineLength + rightLineLength + leftArrowBaseLength + rightArrowBaseLength + leftArrowHeadLength + i)

        lines.InsertNextCell(arrowTipL)
        #lines.InsertNextCell(arrowTipR)


        ############################################################ ADDING COLORS TO CELLS/POINTS ############################################################
        # unfinished

        orange = [255, 165, 0]
        cyan = [0, 255, 255]
        green = [0, 255, 0]
        greenish = [0, 255, 128]
        blue = [0, 0, 255]
        blueish = [0, 128, 255]
        muddycyan = [0, 204, 204]
        pink = [255, 0, 255]
        white = [255, 255, 255]
        black = [0, 0, 0]
        ccolors = vtk.vtkUnsignedCharArray()
        ccolors.SetNumberOfComponents(3)
        ccolors.SetName("Cell Colors")
        #pcolors = vtk.vtkUnsignedCharArray()
        #pcolors.SetNumberOfComponents(3)
        #pcolors.SetName("Point Colors")

        # each cell/line gets its own color
        # center line
        ccolors.InsertNextTuple(pink)
        # bottom arc
        ccolors.InsertNextTuple(muddycyan)
        # left edge
        ccolors.InsertNextTuple(pink)
        # right edge
        ccolors.InsertNextTuple(pink)
        # left arrowbase
        ccolors.InsertNextTuple(muddycyan)
        # right arrowbase
        ccolors.InsertNextTuple(muddycyan)
        # # left arrowtip
        ccolors.InsertNextTuple(orange)
        # # right arrowtip
        ccolors.InsertNextTuple(orange)

        '''
        irrelevant for now
        # alternative: each point gets its own color
        for i in range(linePoints.GetNumberOfPoints()):
            if i < centerLength:
                pcolors.InsertNextTuple(pink)
            elif i < centerLength + lineLength:
                pcolors.InsertNextTuple(cyan)
            elif i < centerLength + lineLength * 2:
                pcolors.InsertNextTuple(cyan)
            else:
                pcolors.InsertNextTuple(cyan)
        '''

        polyOutput.GetCellData().AddArray(ccolors)
        #polyOutput.GetPointData().AddArray(pcolors)

        endTime = time.time()
        print(f"Elapsed time: {endTime - startTime} seconds")
        return 1