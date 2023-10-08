import sys

sys.path.append("../")

from draw import Naki


def test_naki_from_chi_info1():
    pin123chi1 = Naki.from_chi_info(1, 0, 0, False)
    assert pin123chi1.naki_code == 21679
    assert pin123chi1._clean() == pin123chi1.naki_code
    assert pin123chi1.is_chi()
    color, number, which, has_red, exposed, acquired = pin123chi1.pattern_chi()
    assert color == 1
    assert number == 0
    assert which == 0
    assert has_red == False
    assert exposed == [41, 45]
    assert acquired == 37


def test_naki_from_chi_info2():
    sou456chi5red = Naki.from_chi_info(2, 3, 1, True)
    assert sou456chi5red.naki_code == 53391
    assert sou456chi5red._clean() == sou456chi5red.naki_code
    assert sou456chi5red.is_chi()
    color, number, which, has_red, exposed, acquired = sou456chi5red.pattern_chi()
    assert color == 2
    assert number == 3
    assert which == 1
    assert has_red == True
    assert exposed == [85, 93]
    assert acquired == 88


def test_naki_from_chi_info3():
    man567chi7red = Naki.from_chi_info(0, 4, 2, True)
    assert man567chi7red.naki_code == 14503
    assert man567chi7red._clean() == man567chi7red.naki_code
    assert man567chi7red.is_chi()
    color, number, which, has_red, exposed, acquired = man567chi7red.pattern_chi()
    assert color == 0
    assert number == 4
    assert which == 2
    assert has_red == True
    assert exposed == [16, 21]
    assert acquired == 25


def test_naki_from_pon_info1():
    man222pon = Naki.from_pon_info(1, 0, False, 2)
    assert man222pon.naki_code == 1546
    assert man222pon._clean() == man222pon.naki_code
    assert man222pon.is_pon()
    color, number, which, has_red, exposed, acquired = man222pon.pattern_pon()
    assert color == 0
    assert number == 1
    assert which == 0
    assert has_red == False
    assert exposed == [6, 7]
    assert acquired == 5


def test_naki_from_pon_info2():
    pin555pongotred = Naki.from_pon_info(13, 0, True, 3)
    assert pin555pongotred.naki_code == 20075
    assert pin555pongotred._clean() == pin555pongotred.naki_code
    assert pin555pongotred.is_pon()
    color, number, which, has_red, exposed, acquired = pin555pongotred.pattern_pon()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [53, 54]
    assert acquired == 52


def test_naki_from_pon_info3():
    sou555ponhadred = Naki.from_pon_info(22, 1, True, 1)
    assert sou555ponhadred.naki_code == 34409
    assert sou555ponhadred._clean() == sou555ponhadred.naki_code
    assert sou555ponhadred.is_pon()
    color, number, which, has_red, exposed, acquired = sou555ponhadred.pattern_pon()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [88, 90]
    assert acquired == 89


def test_naki_from_pon_info4():
    hatsupon = Naki.from_pon_info(32, 0, False, 1)
    assert hatsupon.naki_code == 49161
    assert hatsupon._clean() == hatsupon.naki_code
    assert hatsupon.is_pon()
    color, number, which, has_red, exposed, acquired = hatsupon.pattern_pon()
    assert color == 3
    assert number == 5
    assert which == 0
    assert has_red == False
    assert exposed == [130, 131]
    assert acquired == 129


def test_naki_from_kakan_info1():
    man444kakan = Naki.from_kakan_info(3, 0)
    assert man444kakan.naki_code == 4656
    assert man444kakan._clean() == man444kakan.naki_code
    assert man444kakan.is_kakan()
    color, number, which, has_red, exposed, acquired = man444kakan.pattern_kakan()
    assert color == 0
    assert number == 3
    assert which == 0
    assert has_red == False
    assert exposed == [13]
    assert acquired is None


def test_naki_from_kakan_info2():
    pin555kakangetred = Naki.from_kakan_info(13, 0)
    assert pin555kakangetred.naki_code == 19984
    assert pin555kakangetred._clean() == pin555kakangetred.naki_code
    assert pin555kakangetred.is_kakan()
    color, number, which, has_red, exposed, acquired = pin555kakangetred.pattern_kakan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [52]
    assert acquired is None


def test_naki_from_kakan_info3():
    sou555kakanhadred = Naki.from_kakan_info(22, 1)
    assert sou555kakanhadred.naki_code == 34352
    assert sou555kakanhadred._clean() == sou555kakanhadred.naki_code
    assert sou555kakanhadred.is_kakan()
    color, number, which, has_red, exposed, acquired = sou555kakanhadred.pattern_kakan()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [89]
    assert acquired == None


def test_naki_from_kakan_info4():
    peikakan = Naki.from_kakan_info(30, 0)
    assert peikakan.naki_code == 46128
    assert peikakan._clean() == peikakan.naki_code
    assert peikakan.is_kakan()
    color, number, which, has_red, exposed, acquired = peikakan.pattern_kakan()
    assert color == 3
    assert number == 3
    assert which == 0
    assert has_red == False
    assert exposed == [121]
    assert acquired == None


def test_naki_from_minkan_info1():
    man777minkan = Naki.from_minkan_info(6, 3, 1)
    assert man777minkan.naki_code == 6403
    assert man777minkan._clean() == man777minkan.naki_code
    assert man777minkan.is_minkan()
    color, number, which, has_red, exposed, acquired = man777minkan.pattern_minkan()
    assert color == 0
    assert number == 6
    assert which == 1
    assert has_red == False
    assert exposed == [24, 26, 27]
    assert acquired == 25


def test_naki_from_minkan_info2():
    pin555minkangetred = Naki.from_minkan_info(13, 3, 0)
    assert pin555minkangetred.naki_code == 13315
    assert pin555minkangetred._clean() == pin555minkangetred.naki_code
    assert pin555minkangetred.is_minkan()
    (
        color,
        number,
        which,
        has_red,
        exposed,
        acquired,
    ) = pin555minkangetred.pattern_minkan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [53, 54, 55]
    assert acquired == 52


def test_naki_from_minkan_info3():
    sou555minkanhadred = Naki.from_minkan_info(22, 1, 1)
    assert sou555minkanhadred.naki_code == 22785
    assert sou555minkanhadred._clean() == sou555minkanhadred.naki_code
    assert sou555minkanhadred.is_minkan()
    (
        color,
        number,
        which,
        has_red,
        exposed,
        acquired,
    ) = sou555minkanhadred.pattern_minkan()
    assert color == 2
    assert number == 4
    assert which == 1
    assert has_red == True
    assert exposed == [88, 90, 91]
    assert acquired == 89


def test_naki_from_minkan_info4():
    nanminkan = Naki.from_minkan_info(28, 3, 1)
    assert nanminkan.naki_code == 28931
    assert nanminkan._clean() == nanminkan.naki_code
    assert nanminkan.is_minkan()
    color, number, which, has_red, exposed, acquired = nanminkan.pattern_minkan()
    assert color == 3
    assert number == 1
    assert which == 1
    assert has_red == False
    assert exposed == [112, 114, 115]
    assert acquired == 113


def test_naki_from_ankan_info1():
    man888ankan = Naki.from_ankan_info(8)
    assert man888ankan.naki_code == 8192
    assert man888ankan._clean() == man888ankan.naki_code
    assert man888ankan.is_ankan()
    color, number, which, has_red, exposed, acquired = man888ankan.pattern_ankan()
    assert color == 0
    assert number == 8
    assert which == 0
    assert has_red == False
    assert exposed == [32, 33, 34, 35]
    assert acquired == None


def test_naki_from_ankan_info2():
    pin555ankangetred = Naki.from_ankan_info(13)
    assert pin555ankangetred.naki_code == 13312
    assert pin555ankangetred._clean() == pin555ankangetred.naki_code
    assert pin555ankangetred.is_ankan()
    color, number, which, has_red, exposed, acquired = pin555ankangetred.pattern_ankan()
    assert color == 1
    assert number == 4
    assert which == 0
    assert has_red == True
    assert exposed == [52, 53, 54, 55]
    assert acquired == None
