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
        self._grid_dims = [1, 1]
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

    @smproperty.intvector(name="Glyph Width", number_of_elements=1, default_values=1)
    @smdomain.intrange(min=0, max=5)
    def SetThickness(self, d):
        self._thickness = d
        self.Modified()

    @smproperty.intvector(name="Glyph Length", number_of_elements=1, default_values=7)
    @smdomain.intrange(min=1, max=10)
    def SetLength(self, l):
        self._length = l
        self.Modified()

    @smproperty.intvector(name="Grid (rows | cols)", number_of_elements=2, default_values=[1,1])
    @smdomain.intrange(min=1,max=10)
    def SetGridDims(self, rows, cols):
        self._grid_dims = [rows, cols]
        self.Modified()

    @smproperty.intvector(name="Steps per One", number_of_elements=1, default_values=2)
    @smdomain.intrange(min=1, max=10)
    def SetGrain(self, d):
        self._steps = d
        self.Modified()

    def FillOutputPortInformation(self, port, info):
        info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkPolyData')
        return 1

    def generate_grid_points(self, bounds):
        """
        Create a grid of uniformly spaced coordinates within the bounding box.
        """
        min_x, max_x, min_y, max_y = bounds[0],bounds[1],bounds[2],bounds[3]
        rows, cols = int(self._grid_dims[0]), int(self._grid_dims[1])
        
        x = np.linspace(min_x + (max_x - min_x) / (cols + 1), max_x - (max_x - min_x) / (cols + 1), cols)
        y = np.linspace(min_y + (max_y - min_y) / (rows + 1), max_y - (max_y - min_y) / (rows + 1), rows)
        
        xv, yv = np.meshgrid(x,y)
        
        grid_points = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(rows * cols)])
        return grid_points


    def findIndex(self, dims, bounds, point):
        # dims = input0.GetDimensions()
        # bounds = input0.GetBounds()

        xdir = np.linspace(bounds[0], bounds[1], dims[0])
        ydir = np.linspace(bounds[2], bounds[3], dims[1])

        index_px = np.searchsorted(xdir, point[0])
        index_py = np.searchsorted(ydir, point[1])

        return [index_px-1, index_py-1]


    def GetInterpVector(self, input0, point):
        # Get essential grid properties
        dimensions = np.array(input0.GetDimensions())
        bounds = np.array(input0.GetBounds())
        spacing = np.array(input0.GetSpacing())
        origin = np.array(input0.GetOrigin())
        
        # Clamp point to be within the bounds
        point = np.array([
            max(bounds[0], min(point[0], bounds[1])),
            max(bounds[2], min(point[1], bounds[3]))
        ])
        
        # Calculate the indices for the corners of the cell containing the point
        indices = np.floor((point - origin[:2]) / spacing[:2]).astype(int)
        indices = np.clip(indices, 0, dimensions[:2] - 2)
        
        # Compute the fractional part within the cell
        t = (point - (origin[:2] + indices * spacing[:2])) / spacing[:2]

        # Retrieve corner values using direct NumPy array access
        def get_value_at_index(idx):
            index_flat = idx[1] * dimensions[0] + idx[0]
            return np.array(input0.GetPointData().GetArray(0).GetTuple(index_flat))

        q11 = get_value_at_index(indices)
        q21 = get_value_at_index(indices + [1, 0])
        q12 = get_value_at_index(indices + [0, 1])
        q22 = get_value_at_index(indices + [1, 1])

        # Perform bilinear interpolation using vectorized operations
        interpolated = (q11 * (1 - t[0]) * (1 - t[1]) +
                        q21 * t[0] * (1 - t[1]) +
                        q12 * (1 - t[0]) * t[1] +
                        q22 * t[0] * t[1])

        # Append zero to the interpolated result to form a 3D vector
        return np.append(interpolated, 0)
