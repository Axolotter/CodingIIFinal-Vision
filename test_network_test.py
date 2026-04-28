from network_test import predictImg
from PIL import Image

def test_pred_no():
    assert predictImg(Image.open("Rebuilt-1/test-quarters/no_fuel/1_quarter_1.png")) == 1
    assert predictImg(Image.open("Rebuilt-1/test-quarters/no_fuel/1_quarter_3.png")) == 1
    assert predictImg(Image.open("Rebuilt-1/test-quarters/no_fuel/2_quarter_1.png")) == 1
    assert predictImg(Image.open("Rebuilt-1/test-quarters/no_fuel/3_quarter_2.png")) == 1
    assert predictImg(Image.open("Rebuilt-1/test-quarters/no_fuel/4_quarter_2.png")) == 1


def test_pred_fuel():
    assert predictImg(Image.open("Rebuilt-1/test-quarters/fuel/0_quarter_3.png")) == 0
    assert predictImg(Image.open("Rebuilt-1/test-quarters/fuel/0_quarter_4.png")) == 0
    assert predictImg(Image.open("Rebuilt-1/test-quarters/fuel/1_quarter_2.png")) == 0
    assert predictImg(Image.open("Rebuilt-1/test-quarters/fuel/2_quarter_3.png")) == 0
    assert predictImg(Image.open("Rebuilt-1/test-quarters/fuel/3_quarter_3.png")) == 0