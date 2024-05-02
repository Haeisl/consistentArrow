from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject
from pyprtl.models.ModelBase import *
#from sympy import symbols, diff
import numpy as np
import vtk

from paraview.util.vtkAlgorithm import smproxy, smproperty

@smproxy.filter(label="tensorFromVector")
@smproperty.input(name="Input")

class tensorFromVector(VTKPythonAlgorithmBase):
    def __init__(self):
        self._script = '_ = (x + y, 2 * y, -z)'
        VTKPythonAlgorithmBase.__init__(self)
    
    @smproperty.stringvector(name='Script', default_values='_ = (x + y, 2 * y, -z)')
    @smhint_widget_multiline(True)
    def SetScript(self, script):
        self._script = script
        self.Modified()

    def FillOutputPortInformation(self, port, info):
        info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkImageData')
        return 1
    
    def RequestData(self, request, inInfo, outInfo):
        
        # get the first input
        input0 = vtkImageData.GetData(inInfo[0])
        
        # get the output
        output = vtkImageData.GetData(outInfo)
        
        # new vtkImageData instance
        myVtkImageData = vtk.vtkImageData()
        myVtkImageData.CopyStructure(input0)
        
        inputArray = input0.GetPointData().GetArray(0)
        numberOfPoints = input0.GetNumberOfPoints()
        dimensions = myVtkImageData.GetDimensions()
        spacing = myVtkImageData.GetSpacing()
        origin = myVtkImageData.GetOrigin()
        
        # create data array to hold the 3x3 tensors/matrices
        tensorVtkDataArray = vtk.vtkDoubleArray()
        tensorVtkDataArray.SetNumberOfComponents(9)
        tensorVtkDataArray.SetNumberOfTuples(numberOfPoints)
        tensorVtkDataArray.SetName("tensors")
        # 68921 tuples atm, 0 to 68920
        
        def insertJacobianAtPoint(idx, px, py, pz):
            # e.g. for _ = (x*x + y, 2 * y*y, -z*z)
            #   -> jacobian in column vectors: (2*x, 0, 0) (1, 4*y, 0) (0, 0, -2*z)
            # InsertTuple9 -> column major
            
            '''
            def findFunctions(string):
                #self._script = '_ = (x*x + y, 2 * y*y, -z*z)'
                numberOfComponents = string.count(',') + 1
                
                indices = [-1 for i in range(numberOfComponents)]
                indices[0] = string.find('(')
                indices[numberOfComponents-1] = string.find(')')
                
                funcs = []
                
                for i in range(1, numberOfComponents):
                    if i != numberOfComponents-1:
                        indices[i] = string.find(',', indices[i-1])
                    
                    funcs.append(parse_expr(string[indices[i-1]:indices[i]]))
                
                return funcs
            
            x, y, z = symbols('x, y, z', real=True)
            f1, f2, f3 = findFunctions(self._script)
            df1dx = diff(f1, x)
            df2dx = diff(f2, x)
            df3dx = diff(f3, x)
            
            df1dy = diff(f1, y)
            df2dy = diff(f2, y)
            df3dy = diff(f3, y)
            
            df1dz = diff(f1, z)
            df2dz = diff(f2, z)
            df3dz = diff(f3, z)
            
            tensorVtkDataArray.InsertTuple9(idx,
                                 df1dx,    df2dx,    df3dx,
                                 df1dy,    df2dy,    df3dy,
                                 df1dz,    df2dz,    df3dz
            )
            '''
            
            tensorVtkDataArray.InsertTuple9(idx,
                                 np.sin(px*py),    0.,    2.,
                                 0.,    np.sin(py*pz),    0.,
                                 2.,    0.,    np.cos(pz*px)
            )
        
        for z in range(dimensions[2]):
            for y in range(dimensions[1]):
                for x in range(dimensions[0]):
                    #coords for our 41 points between [-2,2]
                    xcoord = origin[0] + spacing[0]*x
                    ycoord = origin[1] + spacing[1]*y
                    zcoord = origin[2] + spacing[2]*z
                    idx = x + y * dimensions[0] + z * dimensions[0] * dimensions[1]
                    insertJacobianAtPoint(idx, xcoord, ycoord, zcoord)
        
        myVtkImageData.GetPointData().AddArray(tensorVtkDataArray)
        
        output.ShallowCopy(myVtkImageData)
        
        return 1