from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
from pyprtl.models.ModelBase import *
import numpy as np
import vtk
import time
from functools import partial

from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(name="consistentArrow", label="Field Consistent Arrow Glyph")
@smproperty.input(name="Input")
@smproperty.xml("""<OutputPort name="PolyOutput" index="0" id="port0" />""")
class consistentArrow(VTKPythonAlgorithmBase):
    def __init__(self):
        self._mode = "rk4"
        self._normalize = False
        self._center = [1., 1.]
        self._thickness = 1.
        self._length = 7
        self._stepsize = 0.5
        self._grid_dims = [0, 0]
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

    @smproperty.stringvector(name="StringInfo", information_only="1")
    def GetStrings(self):
        return ["rk4", "euler"]

    @smproperty.stringvector(name="Variant", number_of_elements="1", default="rk4")
    @smdomain.xml("""
        <StringListDomain name="list">
            <RequiredProperties>
                <Property name="StringInfo" function="StringInfo"/>
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetString(self, value):
        self._mode = value
        self.Modified()

    @smproperty.xml("""
        <IntVectorProperty name="normalize"
            label="Normalize"
            command="SetNormalize"
            number_of_elements="1"
            default_values="0">
            <BooleanDomain name="bool" />
        </IntVectorProperty>
    """)
    def SetNormalize(self, val):
        self._normalize = val
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

        # normalize the vector if required
        if self._normalize:
            norm = np.linalg.norm(interpolated)
            if norm > 0:
                interpolated /= norm

        # Append zero to the interpolated result to form a 3D vector
        return np.append(interpolated, 0)


    def get_orthogonal(self, vector):
        # return the orthogonal of a 2D vector (in 3D space, with z component = 0)
        return np.array([vector[1], -vector[0], 0.])


    def rk4_standard(self, image_data, cur, steps, forward, stepsize):
        # runge kutta 4 method for integration in a 2d vector field
        direction_forward = 1 if forward else -1
        t = 0
        points = []

        while t < steps:
            # compute k_1 to k_4 for standard vector field
            k_1 = self.bilinear_interpolation(image_data, cur)

            mid_point_1 = cur + k_1 * 0.5
            k_2 = self.bilinear_interpolation(image_data, mid_point_1)

            mid_point_2 = cur + k_2 * 0.5
            k_3 = self.bilinear_interpolation(image_data, mid_point_2)

            end_point = cur + k_3
            k_4 = self.bilinear_interpolation(image_data, end_point)

            next_point = cur + direction_forward * self._stepsize / 6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += stepsize

        return points


    def rk4_orthogonal(self, image_data, cur, steps, left, stepsize):
        # runge kutta 4 method for integrating the orthogonal of the 2d vector field
        direction_left = -1 if left else 1
        t = 0
        points = []

        while t < steps:
            # compute k_1 to k_4 for orthogonal vector field
            k_1 = self.bilinear_interpolation(image_data, cur)
            k_1 = self.get_orthogonal(k_1) * direction_left
            # k_1 = np.array([direction_left * k_1[1], -direction_left * k_1[0], 0.0]) * self.__scaling

            mid_point_1 = cur + k_1 * 0.5
            k_2 = self.bilinear_interpolation(image_data, mid_point_1)
            k_2 = self.get_orthogonal(k_2) * direction_left

            mid_point_2 = cur + k_2 * 0.5
            k_3 = self.bilinear_interpolation(image_data, mid_point_2)
            k_3 = self.get_orthogonal(k_3) * direction_left

            end_point = cur + k_3
            k_4 = self.bilinear_interpolation(image_data, end_point)
            k_4 = self.get_orthogonal(k_4) * direction_left

            next_point = cur + self._stepsize / 6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += stepsize

        return points


    def euler_standard(self, image_data, cur, steps, forward, stepsize):
        # euler method for integrating a 2d vector field
        direction_forward = 1 if forward else -1
        t = 0
        points = []

        while t < steps:
            vec = self.bilinear_interpolation(image_data, cur)
            next_point = cur + direction_forward * vec
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += stepsize

        return points


    def euler_orthogonal(self, image_data, cur, steps, left, stepsize):
        # euler method for integrating the orthogonal of a 2d vector field
        direction_left = -1 if left else 1
        t = 0
        points = []

        while t < steps:
            vec = self.bilinear_interpolation(image_data, cur)
            vec = self.get_orthogonal(vec) * direction_left
            next_point = cur + vec
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += stepsize

        return points


    def binary_search(self, image_data, start, end, parallel_func, orthogonal_func):
        # first iteration a = 0, b = 1
        # second iteration a = 1, b = 0
        # third to n-th iteration a_n = (a_{n-1} + a_{n-2}) / 2
        points = []
        factor_a_0, factor_a_1 = 0., 1.
        dist_0, dist_1 = np.inf, np.inf

        for i in range(20):
            if i == 0:
                factor_a = factor_a_0
            elif i == 1:
                factor_a = factor_a_1
            else:
                factor_a = (factor_a_0 + factor_a_1) / 2.

            factor_b = 1 - factor_a
            min_dist = np.inf
            cur = start

            while True:
                parallel_vec = parallel_func(cur=cur, steps=1) - start
                orthogonal_vec = orthogonal_func(cur=cur, steps=1) - start
                next = cur + factor_a * parallel_vec + factor_b * orthogonal_vec

                dist = np.linalg.norm(end - next)

                if dist <= 1e-3:
                    points.append(next)
                    return points

                if dist <= min_dist:
                    points.append(next)
                    min_dist = dist
                else:
                    break

                cur = next

            if i == 0:
                dist_0 = min_dist
            elif i == 1:
                dist_1 = min_dist
            else:
                if dist_0 > dist_1:
                    dist_0 = dist_1
                    dist_1 = min_dist
                    factor_a_0 = factor_a_1
                    factor_a_1 = factor_a
                else:
                    dist_1 = min_dist
                    factor_a_1 = factor_a

        return points


    def construct_glyph(self, segments, points, lines):
        start_index = points.GetNumberOfPoints()

        for length, segment_points in segments:
            for p in segment_points:
                points.InsertNextPoint(p)

            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(length)
            for i in range(length):
                polyline.GetPointIds().SetId(i, start_index + i)

            lines.InsertNextCell(polyline)
            start_index += length


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
        integrate_std_fw = partial(getattr(self, f"{self._mode}_standard"), image_data=image_data, forward=True, stepsize=self._stepsize)
        integrate_std_bw = partial(getattr(self, f"{self._mode}_standard"), image_data=image_data, forward=False, stepsize=self._stepsize)
        integrate_orth_l = partial(getattr(self, f"{self._mode}_orthogonal"), image_data=image_data, left=True, stepsize=self._stepsize)
        integrate_orth_r = partial(getattr(self, f"{self._mode}_orthogonal"), image_data=image_data, left=False, stepsize=self._stepsize)
        # TODO: this may change:
        integrate_arrowhead_std = partial(getattr(self, f"{self._mode}_standard"), image_data=image_data, forward=True, stepsize=1)
        integrate_arrowhead_orth_l = partial(getattr(self, f"{self._mode}_orthogonal"), image_data=image_data, left=True, stepsize=1)
        integrate_arrowhead_orth_r = partial(getattr(self, f"{self._mode}_orthogonal"), image_data=image_data, left=False, stepsize=1)
        # TODO end

        # set number of units after which the arrow base starts for left / right lines
        side_line_length = self._length + int(self._length/2)

        origins = np.array([self._center[0], self._center[1], 0.0]).reshape((1,3))
        # print(f"Drawing glyph with midpoint at ({origins[0][0]},{origins[0][1]})")

        # print(f"{self._grid_dims=}")
        if self._grid_dims[0] > 0 and self._grid_dims[1] > 0:
            # print(f"Drawing additional glyphs in a {self._grid_dims[0]}x{self._grid_dims[1]} grid")
            grid_points = self.generate_grid_points(bounds)
            origins = np.concatenate((origins, grid_points), axis=0)

        # self._length          | Glyph center line total length
        # self._stepsize        | Stepsize for integration
        # self._thickness       | Glyph width; unit length of the arcs at the bottom of the arrow
        for ORIGIN in origins:
            # compute the points
            # center line -> left / right arc -> left / right line -> left / right arrowbase -> left / right arrowhead

            center_line_backwards = integrate_std_bw(cur=ORIGIN, steps=self._length)

            center_line_forwards = integrate_std_fw(cur=ORIGIN, steps=self._length)

            # first point of the center line is the last point in the integration starting at origin and going backwards
            bottom_arc_left = integrate_orth_l(cur=center_line_backwards[-1], steps=self._thickness)

            # first point of the center line is the last point in the integration starting at origin and going backwards
            bottom_arc_right = integrate_orth_r(cur=center_line_backwards[-1], steps=self._thickness)

            # first point of left line is last point of left arc
            side_line_left = integrate_std_fw(cur=bottom_arc_left[-1], steps=side_line_length)

            # first point of right line is last point of right arc
            side_line_right = integrate_std_fw(cur=bottom_arc_right[-1], steps=side_line_length)

            # first point of left arrowbase is last point of left line
            arrowbase_left = integrate_orth_l(cur=side_line_left[-1], steps=self._thickness)

            # first point of right arrowbase is last point of right line
            arrowbase_right = integrate_orth_r(cur=side_line_right[-1], steps=self._thickness)


            # TODO: make this better lmao
            # first point of left arrowhead is last point of left arrowbase
            arrowhead_left = self.binary_search(
                image_data=image_data,
                start=arrowbase_left[-1],
                end=center_line_forwards[-1],
                parallel_func=integrate_arrowhead_std,
                orthogonal_func=integrate_arrowhead_orth_r
            )
            # first point of right arrowhead is last point of right arrowbase
            arrowhead_right = self.binary_search(
                image_data=image_data,
                start=arrowbase_right[-1],
                end=center_line_forwards[-1],
                parallel_func=integrate_arrowhead_std,
                orthogonal_func=integrate_arrowhead_orth_l
            )
            # TODO end

            line_lists = [
                center_line_backwards,
                center_line_forwards,
                bottom_arc_left,
                bottom_arc_right,
                side_line_left,
                side_line_right,
                arrowbase_left,
                arrowbase_right,
                arrowhead_left,
                arrowhead_right,
            ]

            line_lengths = [len(lst) for lst in line_lists]

            segments = [
                (line_lengths[0] + line_lengths[1] + 1, center_line_backwards[::-1] + [ORIGIN] + center_line_forwards),
                (line_lengths[2] + line_lengths[3] + 1, bottom_arc_left[::-1] + [center_line_backwards[-1]] + bottom_arc_right),
                (line_lengths[4] + 1, [bottom_arc_left[-1]] + side_line_left),
                (line_lengths[5] + 1, [bottom_arc_right[-1]] + side_line_right),
                (line_lengths[6] + 1, [side_line_left[-1]] + arrowbase_left),
                (line_lengths[7] + 1, [side_line_right[-1]] + arrowbase_right),
                (line_lengths[8] + 2, [arrowbase_left[-1]] + arrowhead_left + [center_line_forwards[-1]]),
                (line_lengths[9] + 2, [arrowbase_right[-1]] + arrowhead_right + [center_line_forwards[-1]])
            ]

            self.construct_glyph(segments, points, lines)

        poly_output.SetPoints(points)
        poly_output.SetLines(lines)

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} s")
        return 1