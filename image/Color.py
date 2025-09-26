# class to express color
# r, g, b, a are 0-255 range

class Color:
    def __init__(self, r: int, g: int, b: int, a=0) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def show(self):
        pass

    @staticmethod
    def generateFromHSV(h: int, s: int, v: int):
        r = 0
        g = 0
        b = 0