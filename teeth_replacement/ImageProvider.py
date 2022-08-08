from enum import Enum, auto
from script_location import script_location
from MouthAnalyzer import TeethColorType
from PIL import Image

_TEETH_IMAGE_PATH = f'{script_location}/images/teeth/'


class TeethImageType(Enum):
    NARROW_WHITE_CLOSED = auto()
    NARROW_WHITE_OPENED = auto()
    NARROW_YELLOW_CLOSED = auto()
    NARROW_YELLOW_OPENED = auto()
    WIDE_WHITE_CLOSED = auto()
    WIDE_WHITE_OPENED = auto()
    WIDE_YELLOW_CLOSED = auto()
    WIDE_YELLOW_OPENED = auto()

    def _image_name(self) -> str:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 'narrow_closed_white.png'
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 'narrow_white_opened.png'
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 'narrow_closed_yellow.png'
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 'narrow_yellow_opened.png'
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return ''
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return ''
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return ''
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return ''

        # return 'client1.png'

        # return 'client2x.png'

        # return 'client-batch-1-2.jpg'

        return 'client-batch-1-4.jpg'

    def image(self) -> Image:
        image_path = _TEETH_IMAGE_PATH + self._image_name()
        return Image.open(image_path).convert('RGBA')

    def intercanine_distance(self) -> int:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 920
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 920
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 920
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 920
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return 0

        # return 700

        # return 550

        # return 435

        # return 435

        return 410

    # height from top of teeth image to top of the two center teeth
    def gum_above_teeth_height(self) -> int:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 160
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 140
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 160
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 140
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return 0

        # return 280

        # return 160

        # return 24

        return 60

    # height from top of the image to gum between the two center teeth
    def gum_full_height(self) -> int:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 255
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 225
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 255
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 225
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return 0

        # return 360

        # return 200

        # return 65

        return 105


    def gum_none_height(self) -> int:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 285
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 255
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 285
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 255
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return 0

        # return 360  # 380

        # return 230

        # return 80

        return 110

    def center_x(self) -> int:
        # if self == TeethImageType.NARROW_WHITE_CLOSED:
        #     return 867
        # elif self == TeethImageType.NARROW_WHITE_OPENED:
        #     return 868
        # elif self == TeethImageType.NARROW_YELLOW_CLOSED:
        #     return 867
        # elif self == TeethImageType.NARROW_YELLOW_OPENED:
        #     return 868
        # elif self == TeethImageType.WIDE_WHITE_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_WHITE_OPENED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_CLOSED:
        #     return 0
        # elif self == TeethImageType.WIDE_YELLOW_OPENED:
        #     return 0

        # return 651

        # return 570

        # return 362

        return 346


class TeethImageProvider:

    # Criteria to pick image:
    # 1. Shape (narrow, wide)
    # 2. Color
    # 3. Opened or not

    @staticmethod
    def image_type(color: TeethColorType, is_narrow: bool = True, is_opened: bool = False) -> TeethImageType:
        if color == TeethColorType.WHITE:
            return TeethImageType.NARROW_WHITE_OPENED if is_opened else TeethImageType.NARROW_WHITE_CLOSED
        else:
            return TeethImageType.NARROW_YELLOW_OPENED if is_opened else TeethImageType.NARROW_YELLOW_CLOSED
