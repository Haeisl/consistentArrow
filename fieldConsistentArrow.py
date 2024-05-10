from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
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


    @smproperty.doublevector(name="Grid Rows | Cols", default_values=[0, 0])
    @smdomain.intrange()
    def SetGridDimensions(self, rows, cols):
        self.grid_dimensions = [rows, cols]
        self.Modified()


    @smproperty.intvector(name="Steps per One", number_of_elements=1, default_values=2)
    @smdomain.intrange(min=1, max=10)
    def SetGrain(self, d):
        self._steps = d
        self.Modified()


    def FillOutputPortInformation(self, port, info):
        info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkPolyData')
        return 1


    def generate_grid_points(self, bounds, cols, rows):
        min_x, max_x, min_y, max_y = bounds
        grid_points = []

        # Compute the spacing between the points
        dx = (max_x - min_x) / (cols + 1)
        dy = (max_y - min_y) / (rows + 1)

        # Generate the grid points
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                x = min_x + j * dx
                y = min_y + i * dy
                grid_points.append((x, y))

        return grid_points
