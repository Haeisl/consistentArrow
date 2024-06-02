from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
from vtk.vtkCommonCore import vtkDoubleArray
from pyprtl.models.ModelBase import *
import numpy as np
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
        self._thickness = 1.
        self._length = 7
        self._stepsize = 0.5
        self._grid_dims = [0, 0]
        self.__scaling = 1/10
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

    @smproperty.stringvector(name="StringInfo", information_only="1")
    def GetStrings(self):
        return ["rk4", "euler"]

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

    @smproperty.doublevector(name="Glyph Width", number_of_elements=1, default_values=1.)
    @smdomain.doublerange(min=0.5, max=5)
    def SetThickness(self, d):
        self._thickness = d
        self.Modified()

    @smproperty.intvector(name="Glyph Length", number_of_elements=1, default_values=7)
    @smdomain.intrange(min=1, max=10)
    def SetLength(self, l):
        self._length = l
        self.Modified()

    @smproperty.intvector(name="Grid [rows | cols]", number_of_elements=2, default_values=[0,0])
    @smdomain.intrange(min=1,max=10)
    def SetGridDims(self, rows, cols):
        self._grid_dims = [rows, cols]
        self.Modified()

    @smproperty.doublevector(name="Stepsize", number_of_elements=1, default_values=0.5)
    @smdomain.doublerange(min=0.1, max=1)
    def SetGrain(self, d):
        self._stepsize = d
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


    def clip_point(self, image_data, point):
        # bounds to np array for easier manipulation
        bounds = np.array(image_data.GetBounds())

        x_bounds = bounds[[0,1]]
        y_bounds = bounds[[2,3]]

        # clip x and y coordinates
        point[0] = np.clip(point[0], x_bounds[0], x_bounds[1])
        point[1] = np.clip(point[1], y_bounds[0], y_bounds[1])

        # set z coordinate to 0
        point[2] = 0.0

        return point


    def bilinear_interpolation(self, image_data, point):
        # Get essential grid properties
        dimensions = np.array(image_data.GetDimensions())
        bounds = np.array(image_data.GetBounds())
        spacing = np.array(image_data.GetSpacing())
        origin = np.array(image_data.GetOrigin())

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
            return np.array(image_data.GetPointData().GetArray(0).GetTuple(index_flat))

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


    def rk4_standard(self, image_data, cur, steps, forward):
        # runge kutta 4 method for integration in a 2d vector field
        direction_forward = 1 if forward else -1
        t = 0
        points = []

        while t < steps - 1e-4:
            # compute k_1 to k_4 for standard vector field
            k_1 = self.bilinear_interpolation(image_data, cur) * self.__scaling

            mid_point_1 = cur + k_1 * 0.5
            k_2 = self.bilinear_interpolation(image_data, mid_point_1) * self.__scaling

            mid_point_2 = cur + k_2 * 0.5
            k_3 = self.bilinear_interpolation(image_data, mid_point_2) * self.__scaling

            end_point = cur + k_3
            k_4 = self.bilinear_interpolation(image_data, end_point) * self.__scaling

            next_point = cur + direction_forward * self._stepsize / 6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += self._stepsize

        return points


    def rk4_orthogonal(self, image_data, cur, steps, left):
        # runge kutta 4 method for integrating the orthogonal of the 2d vector field
        direction_left = -1 if left else 1
        t = 0
        points = []

        while t < steps - 1e-4:
            # compute k_1 to k_4 for orthogonal vector field
            k_1 = self.bilinear_interpolation(image_data, cur)
            k_1 = np.array([direction_left * k_1[1], -direction_left * k_1[0], 0.0]) * self.__scaling

            mid_point_1 = cur + k_1 * 0.5
            k_2 = self.bilinear_interpolation(image_data, mid_point_1)
            k_2 = np.array([direction_left * k_2[1], -direction_left * k_2[0], 0.0]) * self.__scaling

            mid_point_2 = cur + k_2 * 0.5
            k_3 = self.bilinear_interpolation(image_data, mid_point_2)
            k_3 = np.array([direction_left * k_3[1], -direction_left * k_3[0], 0.0]) * self.__scaling

            end_point = cur + k_3
            k_4 = self.bilinear_interpolation(image_data, end_point)
            k_4 = np.array([direction_left * k_4[1], -direction_left * k_4[0], 0.0]) * self.__scaling

            next_point = cur + self._stepsize / 6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += self._stepsize

        return points


    def euler_standard(self, image_data, cur, steps, forward):
        # euler method for integrating a 2d vector field
        direction_forward = 1 if forward else -1
        t = 0
        points = []

        while t < steps - 1e-4:
            vec = self.bilinear_interpolation(image_data, cur) * self.__scaling
            next_point = cur + direction_forward * vec
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += self._stepsize

        return points


    def euler_orthogonal(self, image_data, cur, steps, left):
        # euler method for integrating the orthogonal of a 2d vector field
        direction_left = -1 if left else 1
        t = 0
        points = []

        while t < steps - 1e-4:
            vec = self.bilinear_interpolation(image_data, cur)
            vec = np.array([direction_left * vec[1], -direction_left * vec[0], 0.0]) * self.__scaling
            next_point = cur + direction_left * vec
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += self._stepsize

        return points


    def RequestData(self, request, inInfo, outInfo):
        start_time = time.time()

        # get first input and the output
        image_data = vtkImageData.GetData(inInfo[0])
        poly_output = vtkPolyData.GetData(outInfo)

        # get bounds of underlying vector field
        bounds = image_data.GetBounds()

        # points object to hold all the line points
        points = vtk.vtkPoints()
        # cell array to store the lines' connectivity
        lines = vtk.vtkCellArray()

        # set the integration method
        integrate_standard = getattr(self, f"{self._mode}_standard")
        integrate_orthogonal = getattr(self, f"{self._mode}_orthogonal")

        # set number of unit after which the arrow base starts for left / right lines
        side_line_length = self._length + int(np.floor(self._length/2.))

        origins = np.array([self._center[0], self._center[1], 0.0]).reshape((1,3))
        print(f"Drawing glyph with midpoint at ({origins[0][0]},{origins[0][1]})")

        print(f"{self._grid_dims=}")
        if self._grid_dims[0] > 0 and self._grid_dims[1] > 0:
            print(f"Drawing additional glyphs in a {self._grid_dims[0]}x{self._grid_dims[1]} grid")
            grid_points = self.generate_grid_points(bounds)
            origins = np.concatenate((origins, grid_points), axis=0)

        # self._length          | Glyph center line total length
        # self._stepsize        | Stepsize for integration
        # self._thickness       | Glyph width; unit length of the arcs at the bottom of the arrow

        # standard      (self, image_data, cur, steps, forward)
        # orthogonal    (self, image_data, cur, steps, left
        for ORIGIN in origins:
            # construct the arrows
            # center line -> left / right arc -> left / right line -> left / right arrowbase -> left / right arrowhead
            center_line_backwards = integrate_standard(
                image_data=image_data,
                cur=ORIGIN,
                steps=self._length,
                forward=False
            )
            center_line_forwards = integrate_standard(
                image_data=image_data,
                cur=ORIGIN,
                steps=self._length,
                forward=True
            )
            bottom_arc_left = integrate_orthogonal(
                image_data=image_data,
                cur=center_line_backwards[-1],# first point of the center line is the last point in the integration starting at origin and going backwards
                steps=self._thickness,
                left=True
            )
            bottom_arc_right = integrate_orthogonal(
                image_data=image_data,
                cur=center_line_backwards[-1], # first point of the center line is the last point in the integration starting at origin and going backwards
                steps=self._thickness,
                left=False
            )
            side_line_left = integrate_standard(
                image_data=image_data,
                cur=bottom_arc_left[-1], # first point of left line is last point of left arc
                steps=side_line_length,
                forward=True
            )
            side_line_right = integrate_standard(
                image_data=image_data,
                cur=bottom_arc_right[-1], # first point of right line is last point of right arc
                steps=side_line_length,
                forward=True
            )
            arrowbase_left = integrate_orthogonal(
                image_data=image_data,
                cur=side_line_left[-1], # first point of left arrowbase is last point of left line
                steps=self._thickness,
                left=True
            )
            arrowbase_right = integrate_orthogonal(
                image_data,
                cur=side_line_right[-1], # first point of right arrowbase is last point of right line
                steps=self._thickness,
                left=False
            )

            def construct_glyph(glyph_points, segments, points, lines):
                # Initialize the starting index for glyph_points
                index = 0

                # Prepare the segment data based on the line lengths and corresponding segment points

                # Construct glyph_points and populate the vtkPoints
                for length, segment_points in segments:
                    glyph_points[index:index + length] = segment_points
                    for p in segment_points:
                        points.InsertNextPoint(p)
                    index += length

                # Helper function to create a polyline and set point IDs
                def create_polyline(start, length):
                    polyline = vtk.vtkPolyLine()
                    polyline.GetPointIds().SetNumberOfIds(length)
                    for i in range(length):
                        polyline.GetPointIds().SetId(i, start + i)
                    return polyline

                # Starting index for each polyline
                start_index = 0

                # Create and insert each polyline
                for length, _ in segments:
                    polyline = create_polyline(start_index, length)
                    lines.InsertNextCell(polyline)
                    start_index += length

            line_lengths = [len(lst) for lst in [
                center_line_backwards,
                center_line_forwards,
                bottom_arc_left,
                bottom_arc_right,
                side_line_left,
                side_line_right,
                arrowbase_left,
                arrowbase_right
            ]]
            segments = [
                    (line_lengths[0] + line_lengths[1] + 1, center_line_backwards + [ORIGIN] + center_line_forwards),
                    (line_lengths[2] + line_lengths[3] + 1, bottom_arc_left + [center_line_backwards[-1]] + bottom_arc_right),
                    (line_lengths[4], side_line_left),
                    (line_lengths[5], side_line_right),
                    (line_lengths[6], arrowbase_left),
                    (line_lengths[7], arrowbase_right)
            ]
            number_of_points = sum(line_lengths)
            glyph_points = [None] * (number_of_points + 2)

            construct_glyph(glyph_points, segments, points, lines)

        poly_output.SetPoints(points)
        poly_output.SetLines(lines)
        print(lines.GetNumberOfCells())

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} s")
        return 1