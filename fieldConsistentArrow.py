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

        return np.append(interpolated_vector, 0)
