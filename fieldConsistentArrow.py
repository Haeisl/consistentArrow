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
        breadth = abs(bounds[0]) + abs(bounds[1])
        height = abs(bounds[2]) + abs(bounds[3])

        dist_x = breadth / (cols + 1)
        dist_y = height / (rows + 1)

        center_breadth = breadth / 2
        center_height = height / 2

        if rows % 2 == 1:
            y_components = np.linspace(
                center_height - (rows // 2) * dist_y,
                center_height + (rows // 2) * dist_y,
                rows
            )
        else:
            even_height_lower = center_height - 0.5 * dist_y
            yComponents = np.concatenate(([even_height_lower], [even_height_lower - (i + 1) * dist_y for i in range(rows // 2)],
                                            [even_height_lower + (i + 1) * dist_y for i in range(rows // 2)]))

        if cols % 2 == 1:
            xComponents = np.linspace(center_breadth - (cols // 2) * dist_x,center_breadth + (cols // 2) * dist_x, cols)
        else:
            evenBreadthLower =center_breadth - 0.5 * dist_x
            xComponents = np.concatenate(([evenBreadthLower], [evenBreadthLower - (i + 1) * dist_x for i in range(cols // 2)],
                                            [evenBreadthLower + (i + 1) * dist_x for i in range(cols // 2)]))

        grid = np.column_stack((np.repeat(xComponents, len(yComponents)), np.tile(yComponents, len(xComponents))))

        return grid.tolist()
