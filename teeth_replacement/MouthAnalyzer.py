from enum import Enum, auto
from typing import Optional
from geometry_types import *
from image_operations import *
from dataclasses import dataclass
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc


# use namedtuple?
@dataclass
class _Color:
    r: int
    g: int
    b: int


class _MouthObject(Enum):
    GUM = 'gum'
    TEETH = 'teeth'
    INNER = 'inner'


@dataclass
class _Segment:
    start: Point
    end: Point
    object: _MouthObject

    def height(self) -> int:
        return self.end.y - self.start.y


@dataclass
class _Column:
    _MIN_SEGMENT_HEIGHT_COEFFICIENT = 0.05
    _MIN_MOUTH_OPENED_COEFFICIENT = 0.085

    start: Point
    end: Point
    segments: list

    def _long_segments(self) -> list[_Segment]:
        height = self.end.y - self.start.y

        if height == 0:
            return []
        else:
            min_segment_height = height * _Column._MIN_SEGMENT_HEIGHT_COEFFICIENT
            return list(filter(lambda segment: segment.height() > min_segment_height, self.segments))

    def start_mouth_object(self) -> Optional[_MouthObject]:
        long_segments = self._long_segments()
        return long_segments[0].object if long_segments else None

    def is_opened_mouth(self) -> bool:
        long_segments = self._long_segments()
        teeth_segments = list(filter(lambda segment: segment.object == _MouthObject.TEETH, long_segments))
        teeth_segments = sorted(teeth_segments, key=lambda segment: segment.height())

        if len(teeth_segments) < 2:
            return False

        teeth_segments = teeth_segments[-2:]
        teeth_segments = sorted(teeth_segments, key=lambda segment: segment.start.y)

        teeth_height = teeth_segments[1].end.y - teeth_segments[0].start.y
        teeth_gap = teeth_segments[1].start.y - teeth_segments[0].end.y

        return (teeth_gap / teeth_height) > _Column._MIN_MOUTH_OPENED_COEFFICIENT

    def has_teeth(self) -> bool:
        mouth_objects = list(map(lambda segment: segment.object, self._long_segments()))
        return _MouthObject.TEETH in mouth_objects


class TeethColorType(Enum):
    YELLOW = auto()
    WHITE = auto()


class GumType(Enum):
    NONE = auto()
    SLIGHT = auto()
    GUMMY = auto()


@dataclass
class GumData:
    type: GumType
    height: Optional[int]


class MouthAnalyzer:

    _NO_GUM_MIN_TEETH_AMOUNT = 0.85
    _SLIGHT_GUM_MIN_TEETH_AMOUNT = 0.3  # might lower a bit to 0.28

    _mouth_colors_palette = {
        'gum': _Color(200, 100, 140),
        'teeth': _Color(180, 160, 130),
        'white_teeth': _Color(255, 255, 255),
        'inner_mouth': _Color(20, 20, 20)
    }

    _mouth_image: Image = None
    _columns: list[_Column] = []

    def _color_diff(self, color1: _Color, color2: _Color) -> float:
        color1_rgb = sRGBColor(color1.r, color1.g, color1.b)
        color2_rgb = sRGBColor(color2.r, color2.g, color2.b)

        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)

        return delta_e_cie2000(color1_lab, color2_lab)
        # return delta_e_cmc(color1_lab, color2_lab)

    def _color_type(self, color: _Color, palette) -> str:
        # manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
        # distances = {k: manhattan(v, color) for k, v in palette.items()}
        distances = {k: self._color_diff(v, color) for k, v in palette.items()}
        return min(distances, key=distances.get)

    def _mouth_object_of_color(self, color: _Color, palette) -> _MouthObject:
        color_type = self._color_type(color, palette)

        for object_type in _MouthObject:
            if object_type.value in color_type:
                return object_type

        return _MouthObject.GUM

    def _column_representation(self, x_range) -> list[_Column]:
        pixels = self._mouth_image.load()
        columns = list()

        for x in x_range:
            column_start = None
            column_end = None

            column_segments = list()

            segment_start = None
            segment_end = None
            segment_object = None

            for y in range(self._mouth_image.size[1]):
                pixel = pixels[x, y]

                if pixel[3] == 0 and column_start:  # transparent pixel: column ended, go to next ones
                    column_end = Point(x, y - 1)

                    if segment_start:
                        segment_end = Point(x, y - 1)
                        segment = _Segment(segment_start, segment_end, segment_object)
                        column_segments.append(segment)

                    break

                elif pixel[3] != 0:  # colored pixels

                    #  start a column if hasn't already
                    if not column_start:
                        column_start = Point(x, y)

                    color = _Color(pixel[0], pixel[1], pixel[2])
                    mouth_object = self._mouth_object_of_color(color, MouthAnalyzer._mouth_colors_palette)

                    if mouth_object != segment_object and not segment_start:  # white pixel, start segment if haven't already
                        segment_start = Point(x, y)
                        segment_object = mouth_object
                    elif mouth_object != segment_object and segment_start:  # black pixel, end segment if have started
                        segment_end = Point(x, y - 1)

                        segment = _Segment(segment_start, segment_end, segment_object)
                        column_segments.append(segment)

                        segment_start = Point(x, y)
                        segment_end = None
                        segment_object = mouth_object

            #  create a column
            if column_start and column_end and column_segments:
                column = _Column(column_start, column_end, column_segments)
                columns.append(column)

        return columns

    def process(self, image: Image, mouth_polygon: list[Point], x_range: range) -> None:
        self._mouth_image = cropped_rgba_image(image, mouth_polygon)
        self._columns = self._column_representation(x_range)

    def gum_data(self) -> GumData:
        # get only cols with teeth
        teeth_columns = list(filter(lambda col: col.start_mouth_object() == _MouthObject.TEETH, self._columns))

        # 3 cases:
        # almost all cols are teeth: no gums at all, put the teeth higher
        # most of the cols are teeth: teeth are fully visible (tall teeth) and gums between teeth is a bit visible
        # almost all cols are gum: calculate gum height

        teeth_amount = len(teeth_columns) / len(self._columns)
        gum_type: GumType

        if teeth_amount > MouthAnalyzer._NO_GUM_MIN_TEETH_AMOUNT:
            return GumData(GumType.NONE, None)  # put teeth higher, come up with a better value/indication
        elif MouthAnalyzer._SLIGHT_GUM_MIN_TEETH_AMOUNT < teeth_amount < MouthAnalyzer._NO_GUM_MIN_TEETH_AMOUNT:
            gum_type = GumType.SLIGHT
        else:
            gum_type = GumType.GUMMY

        # get cols only with gum
        gum_columns = list(filter(lambda col: col.start_mouth_object() == _MouthObject.GUM, self._columns))
        gum_heights = list(map(lambda col: col.segments[0].end.y - col.segments[0].start.y, gum_columns))
        height = int(sum(gum_heights) / len(gum_heights))

        return GumData(gum_type, height)

    def teeth_color(self) -> TeethColorType:
        binary = binary_image(self._mouth_image, alpha_channel=True)

        binary_pixels = binary.load()
        mouth_pixels = self._mouth_image.load()

        yellow = 0
        white = 0

        for x in range(binary.size[0]):
            for y in range(binary.size[1]):
                if binary_pixels[x, y] == (255, 255):
                    pixel = mouth_pixels[x, y]
                    color = _Color(pixel[0], pixel[1], pixel[2])
                    color_t = self._color_type(color, MouthAnalyzer._mouth_colors_palette)

                    # print(f'color type at {x} {y}: {color_t}')

                    if color_t == 'white_teeth':
                        white += 1
                    elif color_t == 'teeth':
                        yellow += 1

        teeth_color = TeethColorType.WHITE if white > yellow else TeethColorType.YELLOW
        # print(f'teeth color: {teeth_color} with white count: {white}, yellow count: {yellow}')
        return teeth_color

    def is_opened_mouth(self) -> bool:
        opened_count = 0
        closed_count = 0

        for column in self._columns:
            if column.is_opened_mouth():
                opened_count += 1
            else:
                closed_count += 1

        return opened_count > closed_count

    def is_narrow_jaw(self, left_x_range, right_x_range) -> bool:
        left_columns = self._column_representation(left_x_range)
        teeth_cols_count = 0

        for column in left_columns:
            if column.has_teeth():
                teeth_cols_count += 1

        teeth_cols_ratio = teeth_cols_count / len(left_columns)

        if teeth_cols_ratio > 0.5:  # 0.5 to constant
            return False

        right_columns = self._column_representation(right_x_range)
        teeth_cols_count = 0

        for column in right_columns:
            if column.has_teeth():
                teeth_cols_count += 1

        teeth_cols_ratio = teeth_cols_count / len(right_columns)
        return teeth_cols_ratio < 0.5  # 0.5 to constant
