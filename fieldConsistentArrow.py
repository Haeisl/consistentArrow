import time
from functools import partial

import numpy as np

from vtk import vtkPolyLine, vtkPoints, vtkCellArray
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkDataObject, vtkPolyData
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(name="consistentArrow", label="Field Consistent Arrow Glyph")
@smproperty.input(name="Input")
@smproperty.xml("""<OutputPort name="PolyOutput" index="0" id="port0" />""")
class consistentArrow(VTKPythonAlgorithmBase):
    def __init__(self):
        # ---default---
        self._mode = "rk4"
        self._normalize = True
        self._center = [1., 1.]
        self._grid_dims = [0, 0]
        # ---advanced---
        self._stepsize = 0.5
        self._scaling = 1.
        self._length = 5.
        self._thickness = 1.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)

    @smproperty.stringvector(name="StringInfo", information_only="1")
    def GetVariants(self):
        return ["rk4", "euler"]

    @smproperty.stringvector(name="Variant", number_of_elements="1", default="rk4")
    @smdomain.xml("""
        <StringListDomain name="list">
            <RequiredProperties>
                <Property name="StringInfo" function="StringInfo"/>
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetVariant(self, value):
        self._mode = value
        self.Modified()

    @smproperty.xml(
        """
        <IntVectorProperty name="Normalize"
            label="Normalize"
            command="SetNormalize"
            number_of_elements="1"
            default_values="1">
            <BooleanDomain name="bool" />
        </IntVectorProperty>
        """
    )
    def SetNormalize(self, val):
        self._normalize = val
        self.Modified()

    @smproperty.doublevector(name="Center Point", default_values=[1., 1.])
    @smdomain.doublerange()
    def SetStartPoint(self, x, y):
        self._center = [x, y]
        self.Modified()

    @smproperty.intvector(name="Grid [rows | cols]", number_of_elements=2, default_values=[0,0])
    @smdomain.intrange(min=1,max=10)
    def SetGridDims(self, rows, cols):
        self._grid_dims = [rows, cols]
        self.Modified()

    @smproperty.doublevector(name="Stepsize", number_of_elements=1, default_values=0.5)
    @smdomain.doublerange()
    @smproperty.panel_visibility("advanced")
    def SetGrain(self, d):
        self._stepsize = d
        self.Modified()

    @smproperty.doublevector(name="Glyph scaling", number_of_elements=1, default_values=1.)
    @smdomain.doublerange(min=0.1, max=2)
    @smproperty.panel_visibility("advanced")
    def SetScaling(self, s):
        self._scaling = s
        self.Modified()

    @smproperty.doublevector(name="Glyph Length", number_of_elements=1, default_values=5.)
    @smdomain.doublerange()
    @smproperty.panel_visibility("advanced")
    def SetLength(self, l):
        self._length = l
        self.Modified()

    @smproperty.doublevector(name="Glyph Width", number_of_elements=1, default_values=1.)
    @smdomain.doublerange()
    @smproperty.panel_visibility("advanced")
    def SetThickness(self, d):
        self._thickness = d
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
        """
        Clip given point to edges of image data's bounding box.
        """
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
        spacing = np.array(image_data.GetSpacing())
        origin = np.array(image_data.GetOrigin())

        # Clip point to be within the bounds
        point = self.clip_point(image_data, point)

        # Calculate the indices for the corners of the cell containing the point
        indices = np.floor((point[:2] - origin[:2]) / spacing[:2]).astype(int)
        indices = np.clip(indices, 0, dimensions[:2] - 2)

        # Compute the fractional part within the cell
        t = (point[:2] - (origin[:2] + indices * spacing[:2])) / spacing[:2]

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

        # Normalize the vector if required
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

            vec = direction_forward * self._stepsize / 6. * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = cur + vec * self._scaling
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

            mid_point_1 = cur + k_1 * 0.5
            k_2 = self.bilinear_interpolation(image_data, mid_point_1)
            k_2 = self.get_orthogonal(k_2) * direction_left

            mid_point_2 = cur + k_2 * 0.5
            k_3 = self.bilinear_interpolation(image_data, mid_point_2)
            k_3 = self.get_orthogonal(k_3) * direction_left

            end_point = cur + k_3
            k_4 = self.bilinear_interpolation(image_data, end_point)
            k_4 = self.get_orthogonal(k_4) * direction_left

            vec = self._stepsize / 6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            next_point = cur + vec * self._scaling
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
            next_point = cur + direction_forward * vec * self._scaling
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
            vec = self.get_orthogonal(vec) * direction_left * self._scaling
            next_point = cur + vec
            next_point = self.clip_point(image_data, next_point)
            points.append(next_point)
            cur = next_point
            t += stepsize

        return points


    def binary_search(self, start, end, parallel_func, orthogonal_func):
        # next = cur + a*vec_parallel + b*vec_orthogonal
        # a + b = 1
        # first iteration a = 0, b = 1
        # second iteration a = 1, b = 0
        # third to n-th iteration a_n = (a_{n-1} + a_{n-2}) / 2
        factor_a_0, factor_a_1 = 0., 1.
        dist_0, dist_1 = np.inf, np.inf
        for i in range(20):
            points = []
            if i == 0:
                factor_a = factor_a_0
            elif i == 1:
                factor_a = factor_a_1
            else:
                factor_a = (factor_a_0 + factor_a_1) / 2.

            factor_b = 1 - factor_a
            min_dist = np.inf
            cur = start

            for _ in range(1000):
                parallel_vec = parallel_func(cur=cur, steps=1)[0] - cur
                orthogonal_vec = orthogonal_func(cur=cur, steps=1)[0] - cur
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
            elif abs(dist_0 - min_dist) <= 1e-3 or abs(dist_1 - min_dist) <= 1e-3:
                return points
            else:
                distance_factors = [(dist_0, factor_a_0),(dist_1, factor_a_1),(min_dist, factor_a)]
                distance_factors.sort(key=lambda x: x[0])
                (dist_0, factor_a_0), (dist_1, factor_a_1) = distance_factors[:2]

        return points


    def compute_line_points(self, image_data, origin):
        # Constants
        stepsize = self._stepsize
        thickness = self._thickness
        length = self._length
        side_line_length = length * 1.5

        # Integration functions
        standard = getattr(self, f"{self._mode}_standard")
        orthogonal = getattr(self, f"{self._mode}_orthogonal")

        # Partial functions
        integrate_standard = partial(standard, image_data=image_data, stepsize=stepsize)
        integrate_orthogonal = partial(orthogonal, image_data=image_data, stepsize=stepsize)

        arrowhead_standard = partial(standard, image_data=image_data, forward=True, stepsize=1)
        arrowhead_orthogonal_l = partial(orthogonal, image_data=image_data, left=True, stepsize=1)
        arrowhead_orthogonal_r = partial(orthogonal, image_data=image_data, left=False, stepsize=1)

        # Compute center lines
        center_line_backwards = integrate_standard(cur=origin, forward=False, steps=length)
        center_line_forwards = integrate_standard(cur=origin, forward=True, steps=length)
        # Compute bottom arcs
        bottom_arc_left = integrate_orthogonal(cur=center_line_backwards[-1], left=True, steps=thickness)
        bottom_arc_right = integrate_orthogonal(cur=center_line_backwards[-1], left=False, steps=thickness)
        # Compute side lines
        side_line_left = integrate_standard(cur=bottom_arc_left[-1], forward=True, steps=side_line_length)
        side_line_right = integrate_standard(cur=bottom_arc_right[-1], forward=True, steps=side_line_length)
        # Compute arrow bases
        arrowbase_left = integrate_orthogonal(cur=side_line_left[-1], left=True, steps=thickness)
        arrowbase_right = integrate_orthogonal(cur=side_line_right[-1], left=False, steps=thickness)

        # Compute arrow heads
        arrowhead_left = self.binary_search(
            start=arrowbase_left[-1],
            end=center_line_forwards[-1],
            parallel_func=arrowhead_standard,
            orthogonal_func=arrowhead_orthogonal_r
        )
        arrowhead_right = self.binary_search(
            start=arrowbase_right[-1],
            end=center_line_forwards[-1],
            parallel_func=arrowhead_standard,
            orthogonal_func=arrowhead_orthogonal_l
        )

        point_lists = [
            center_line_backwards, center_line_forwards,
            bottom_arc_left, bottom_arc_right,
            side_line_left, side_line_right,
            arrowbase_left, arrowbase_right,
            arrowhead_left, arrowhead_right,
        ]

        return point_lists


    def construct_glyph(self, segments, points, lines):
        start_index = points.GetNumberOfPoints()

        for length, segment_points in segments:
            for p in segment_points:
                points.InsertNextPoint(p)

            polyline = vtkPolyLine()
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
        points = vtkPoints()
        # cell array to store the lines' connectivity
        lines = vtkCellArray()

        origins = np.array([self._center[0], self._center[1], 0.0]).reshape((1,3))

        if self._grid_dims[0] > 0 and self._grid_dims[1] > 0:
            grid_points = self.generate_grid_points(bounds)
            origins = np.concatenate((origins, grid_points), axis=0)

        for ORIGIN in origins:
            point_lists = self.compute_line_points(image_data=image_data, origin=ORIGIN)

            line_lengths = [len(lst) for lst in point_lists]

            # point_lists =
            #     0: center_line_backwards,     1:center_line_forwards,
            #     2: bottom_arc_left,           3: bottom_arc_right,
            #     4: side_line_left,            5: side_line_right,
            #     6: arrowbase_left,            7: arrowbase_right,
            #     8: arrowhead_left,            9: arrowhead_right
            segments = [
                (line_lengths[0] + line_lengths[1] + 1, point_lists[0][::-1] + [ORIGIN] + point_lists[1]),
                (line_lengths[2] + line_lengths[3] + 1, point_lists[2][::-1] + [point_lists[0][-1]] + point_lists[3]),
                (line_lengths[4] + 1, [point_lists[2][-1]] + point_lists[4]),
                (line_lengths[5] + 1, [point_lists[3][-1]] + point_lists[5]),
                (line_lengths[6] + 1, [point_lists[4][-1]] + point_lists[6]),
                (line_lengths[7] + 1, [point_lists[5][-1]] + point_lists[7]),
                (line_lengths[8] + 2, [point_lists[6][-1]] + point_lists[8] + [point_lists[1][-1]]),
                (line_lengths[9] + 2, [point_lists[7][-1]] + point_lists[9] + [point_lists[1][-1]])
            ]

            self.construct_glyph(segments, points, lines)

        poly_output.SetPoints(points)
        poly_output.SetLines(lines)

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} s")
        return 1