#!/usr/bin/env python

import os

# from cell2mol.module1 import test_addone, test_subtwo
from cell2mol.module1 import addone, subtwo
from cell2mol.cell2mol import cell2mol, split_infofile


def test_modules1():
    assert addone(10) == 11
    assert subtwo(8) == 6


def test_cell2mol():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    infofile = "YOXKUS.info"
    infopath = dir_path + "/infodata/" + infofile
    refcode = split_infofile(infofile)
    cell = cell2mol(infopath, refcode)
    print(
        "=====================================pytest====================================="
    )
    print(cell.version)
    print(cell.refcode)
    print(cell.cellvec)
    print(cell.cellparam)
    print(cell.labels)
    print(cell.pos)

    return cell

def test_check_cell (cell):
    
    assert cell.version == "V16"
    assert cell.refcode == "YOXKUS"
    assert cell.cellvec == [
        [15.136, 0.0, 0.0],
        [0.0, 8.798, 0.0],
        [-4.1830788, 0.0, 17.4721325],
    ]
    assert cell.cellparam == [15.136, 8.798, 17.9659, 90.0, 103.464, 90.0]
    assert cell.labels == [
        "Re",
        "Re",
        "Re",
        "Re",
        "I",
        "I",
        "I",
        "I",
        "I",
        "I",
        "I",
        "I",
        "P",
        "P",
        "P",
        "P",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "C",
        "C",
        "C",
        "O",
        "O",
        "O",
        "O",
    ]
    assert cell.pos == [
        [11.8345549, 8.3225561, 13.3768394],
        [6.3580944, 4.8744439, 4.6407731],
        [4.5948268, 3.9235561, 12.8313594],
        [-0.8816338, 0.4754439, 4.0952931],
        [-1.8756081, 1.8547944, 13.9273863],
        [7.7839313, 2.5442056, 5.19132],
        [3.1689899, 6.2537944, 12.2808125],
        [12.8285293, 6.9432056, 3.5447462],
        [9.8287594, 1.3694087, 12.8007832],
        [4.3522988, 3.0295913, 4.0647169],
        [6.6006224, 5.7684087, 13.4074156],
        [1.1241618, 7.4285913, 4.6713493],
        [-1.7123187, 7.2609894, 14.8878294],
        [7.9472207, 5.9360106, 6.1517631],
        [3.0057004, 2.8619894, 11.3203694],
        [12.6652399, 1.5370106, 2.5843031],
        [-2.0649965, 6.6530476, 12.494322],
        [7.5945429, 6.5439524, 3.7582557],
        [3.3583782, 2.2540476, 13.7138768],
        [13.0179176, 2.1449524, 4.9778106],
        [11.7101618, 6.2826518, 12.3999724],
        [6.2337012, 6.9143482, 3.6639062],
        [4.71922, 1.8836518, 13.8082263],
        [-0.7572406, 2.5153482, 5.0721601],
        [11.1240978, 7.2389944, 11.5351019],
        [5.6476372, 5.9580056, 2.7990356],
        [5.305284, 2.8399944, 14.6730969],
        [-0.1711766, 1.5590056, 5.9370306],
        [12.1081369, 8.1707026, 11.0790792],
        [6.6316763, 5.0262974, 2.343013],
        [4.3207203, 3.7717026, 15.1313106],
        [-1.1552157, 0.6272974, 6.3930533],
        [-1.8023107, 7.8205422, 11.6626484],
        [7.8572287, 5.3764578, 2.9265822],
        [3.0956925, 3.4215422, 14.5455503],
        [12.7552319, 0.9774578, 5.8094841],
        [11.0771345, 5.0421338, 12.945103],
        [5.6006739, 8.1548662, 4.2090367],
        [5.3522473, 0.6431338, 13.2630958],
        [-0.1242133, 3.7558662, 4.5270295],
        [11.0124585, 4.3840434, 12.2479649],
        [5.535998, 0.0149566, 3.5118986],
        [5.4169232, 8.7830434, 13.9602339],
        [-0.0595374, 4.4139566, 5.2241676],
        [10.2008254, 5.248007, 13.2735791],
        [4.7243648, 7.948993, 4.5375128],
        [6.2285563, 0.849007, 12.9346197],
        [0.7520958, 3.549993, 4.1985534],
        [11.6155067, 4.6972522, 13.6614604],
        [6.1390461, 8.4997478, 4.9253942],
        [4.8138751, 0.2982522, 12.5467384],
        [-0.6625855, 4.1007478, 3.8106721],
        [9.720656, 7.099986, 10.9864769],
        [4.2441954, 6.097014, 2.2504107],
        [6.7087257, 2.700986, 15.2217218],
        [1.2322651, 1.698014, 6.4856556],
        [9.7265635, 6.4876452, 10.2474057],
        [4.2501029, 6.7093548, 1.5113395],
        [6.7028183, 2.0886452, 15.960793],
        [1.2263577, 2.3103548, 7.2247268],
        [9.4068957, 7.9569112, 10.6911979],
        [3.9304351, 5.2400888, 1.9551316],
        [7.0224861, 3.5579112, 15.5170009],
        [1.5460255, 0.8410888, 6.7809346],
        [9.1411163, 6.7665418, 11.6748789],
        [3.6646557, 6.4304582, 2.9388127],
        [7.2882654, 2.3675418, 14.5333198],
        [1.8118048, 2.0314582, 5.7972536],
        [11.8802846, 0.4416596, 10.0709372],
        [6.403824, 3.9573404, 1.3348709],
        [4.5490972, 4.8406596, 16.1372616],
        [-0.9273634, 8.3563404, 7.4011953],
        [11.8094225, 0.0492688, 9.1973306],
        [6.3329619, 4.3497312, 0.4612643],
        [4.6199593, 4.4482688, 17.0108682],
        [-0.8565013, 8.7487312, 8.274802],
        [12.6151566, 1.0583994, 10.0866621],
        [7.138696, 3.3406006, 1.3505958],
        [3.8142252, 5.4573994, 16.1215367],
        [-1.6622354, 7.7396006, 7.3854704],
        [11.0676515, 0.9088334, 10.2788556],
        [5.5911909, 3.4901666, 1.5427893],
        [5.3617302, 5.3078334, 15.9293432],
        [-0.1147304, 7.8891666, 7.193277],
        [-0.4257597, 8.4170466, 11.4005665],
        [9.2337797, 4.7799534, 2.6645002],
        [1.7191415, 4.0180466, 14.8076323],
        [11.3786809, 0.3809534, 6.071566],
        [-0.000387, 7.9393152, 10.6859562],
        [9.6591524, 5.2576848, 1.94989],
        [1.2937688, 3.5403152, 15.5222425],
        [10.9533082, 0.8586848, 6.7861763],
        [0.1089246, 8.3475424, 12.1955485],
        [9.768464, 4.8494576, 3.4594822],
        [1.1844572, 3.9485424, 14.0126503],
        [10.8439966, 0.4504576, 5.276584],
        [-0.5189749, 0.5437164, 11.1577038],
        [9.1405645, 3.8552836, 2.4216376],
        [1.8123567, 4.9427164, 15.0504949],
        [11.4718961, 8.2542836, 6.3144287],
        [-1.0960802, 6.1577202, 13.5618693],
        [8.5634592, 7.0392798, 4.825803],
        [2.3894619, 1.7587202, 12.6463295],
        [12.0490014, 2.6402798, 3.9102633],
        [-0.1694483, 6.330161, 13.3329843],
        [9.4900912, 6.866839, 4.5969181],
        [1.46283, 1.931161, 12.8752144],
        [11.1223694, 2.467839, 4.1391482],
        [-1.2204435, 5.2189736, 13.7715348],
        [8.4390959, 7.9780264, 5.0354686],
        [2.5138253, 0.8199736, 12.4366639],
        [12.1733647, 3.5790264, 3.7005977],
        [-0.2678924, 8.1196742, 15.6218337],
        [9.391647, 5.0773258, 6.8857674],
        [1.5612741, 3.7206742, 10.5863651],
        [11.2208136, 0.6783258, 1.8502988],
        [0.9688189, 8.0272952, 15.1535805],
        [10.6283583, 5.1697048, 6.4175143],
        [0.3245628, 3.6282952, 11.0546182],
        [9.9841022, 0.7707048, 2.318552],
        [1.1268239, 7.5082132, 14.3987844],
        [10.7863634, 5.6887868, 5.6627181],
        [0.1665578, 3.1092132, 11.8094144],
        [9.8260972, 1.2897868, 3.0733481],
        [2.0327738, 8.679227, 15.7546219],
        [11.6923132, 4.517773, 7.0185556],
        [-0.7393921, 4.280227, 10.4535769],
        [8.9201474, 0.118773, 1.7175106],
        [2.8956201, 8.5903672, 15.4191569],
        [12.5551595, 4.6066328, 6.6830907],
        [-1.6022384, 4.1913672, 10.7890418],
        [8.057301, 0.2076328, 2.0529756],
        [1.7854712, 0.6624894, 16.8571134],
        [11.4450106, 3.7365106, 8.1210472],
        [-0.4920894, 5.0614894, 9.3510853],
        [9.16745, 8.1355106, 0.6150191],
        [2.4829647, 1.1375814, 17.2502364],
        [12.1425041, 3.2614186, 8.5141702],
        [-1.1895829, 5.5365814, 8.9579623],
        [8.4699565, 7.6604186, 0.2218961],
        [0.5426833, 0.7487098, 17.3760358],
        [10.2022227, 3.6502902, 8.6399695],
        [0.7506985, 5.1477098, 8.832163],
        [10.4102379, 8.0492902, 0.0960967],
        [4.5763598, 1.240518, 0.679666],
        [10.0528204, 3.158482, 9.4157322],
        [0.9001008, 5.639518, 8.0564003],
        [6.3765613, 7.557482, 16.7924666],
        [-0.514997, 0.101177, 16.7487862],
        [9.1445425, 4.297823, 8.01272],
        [1.8083787, 4.500177, 9.4594125],
        [11.4679181, 8.696823, 0.7233463],
        [-1.3811292, 0.1882772, 17.0790095],
        [8.2784103, 4.2107228, 8.3429433],
        [2.6745109, 4.5872772, 9.1291892],
        [12.3340503, 8.6097228, 0.393123],
        [-2.262626, 6.2149072, 16.2595665],
        [7.3969135, 6.9820928, 7.5235003],
        [3.5560077, 1.8159072, 9.9486322],
        [13.2155471, 2.5830928, 1.212566],
        [-1.790928, 4.944476, 16.4325406],
        [7.8686114, 8.252524, 7.6964744],
        [3.0843098, 0.545476, 9.7756581],
        [12.7438492, 3.853524, 1.0395919],
        [-1.246581, 4.5723206, 15.7773357],
        [8.4129584, 8.6246794, 7.0412694],
        [2.5399627, 0.1733206, 10.4308631],
        [12.1995022, 4.2256794, 1.6947969],
        [2.0884266, 4.2045642, 0.0646469],
        [7.5648872, 0.1944358, 8.8007131],
        [3.388034, 8.6035642, 8.6714194],
        [8.8644946, 4.5934358, 17.4074856],
        [2.4087482, 3.3353218, 0.155502],
        [7.8852088, 1.0636782, 8.8915682],
        [3.0677124, 7.7343218, 8.5805643],
        [8.544173, 5.4626782, 17.3166305],
        [1.3001966, 4.7676362, 1.0430863],
        [6.7766572, 8.4293638, 9.7791526],
        [4.176264, 0.3686362, 7.6929799],
        [9.6527246, 4.0303638, 16.4290462],
        [1.0738209, 4.26703, 1.7926408],
        [6.5502815, 0.13197, 10.528707],
        [4.4026397, 8.66603, 6.9434255],
        [9.8791003, 4.53097, 15.6794917],
        [0.8438712, 6.053024, 0.926023],
        [6.3203318, 7.143976, 9.6620893],
        [4.6325894, 1.654024, 7.8100432],
        [10.10905, 2.744976, 16.5461095],
        [0.3382392, 6.4295784, 1.6091834],
        [5.8146998, 6.7674216, 10.3452497],
        [5.1382213, 2.0305784, 7.1268828],
        [10.6146819, 2.3684216, 15.8629491],
        [-3.0465354, 6.7929358, 17.2642141],
        [6.613004, 6.4040642, 8.5281479],
        [4.3399171, 2.3939358, 8.9439846],
        [13.9994566, 2.0050642, 0.2079184],
        [-3.3679523, 7.6612984, 17.1716118],
        [6.2915871, 5.5357016, 8.4355456],
        [4.661334, 3.2622984, 9.0365869],
        [14.3208734, 1.1367016, 0.3005207],
        [10.5260703, 7.9076424, 14.6398998],
        [5.0496097, 5.2893576, 5.9038336],
        [5.9033115, 3.5086424, 11.5682989],
        [0.4268509, 0.8903576, 2.8322327],
        [9.6886243, 7.6164286, 15.2549189],
        [4.2121637, 5.5805714, 6.5188526],
        [6.7407575, 3.2174286, 10.9532799],
        [1.2642969, 1.1815714, 2.2172136],
    ]
    assert cell.warning_list == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

def test_check_cellrefmoleclist (cell):    
  for mol in cell.refmoleclist:
      assert mol.version == "V16"
      assert mol.refcode == "YOXKUS"
      assert mol.name == "YOXKUS_Reference_0"
      assert mol.atlist == [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17,
          18,
          19,
          20,
          21,
          22,
          23,
          24,
          25,
          26,
          27,
          28,
          29,
          30,
          31,
          32,
          33,
          34,
          35,
          36,
          37,
          38,
          39,
          40,
          41,
          42,
          43,
          44,
          45,
          46,
          47,
          48,
          49,
          50,
          51,
      ]
      assert mol.labels == [
          "Re",
          "I",
          "I",
          "P",
          "C",
          "C",
          "C",
          "C",
          "C",
          "C",
          "H",
          "H",
          "H",
          "C",
          "H",
          "H",
          "H",
          "C",
          "H",
          "H",
          "H",
          "C",
          "H",
          "H",
          "H",
          "C",
          "H",
          "H",
          "C",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "H",
          "C",
          "O",
      ]
      assert mol.coord == [
          [-3.3014450586739708, 8.322556080000002, 13.376839369067877],
          [-1.8756081124277317, 10.652794360000001, 13.927386264379235],
          [-5.307240629836729, 10.167408700000003, 12.800783160295566],
          [-1.7123186753996211, 7.2609894000000015, 14.887829388316568],
          [-2.064996467258469, 6.653047600000001, 12.494321956113998],
          [-3.425838241607237, 6.282651800000001, 12.399972440573494],
          [-4.0119022398042805, 7.238994400000001, 11.535101881452192],
          [-3.027863082489995, 8.1707026, 11.079079223006413],
          [-1.8023107152216866, 7.820542200000001, 11.662648448756949],
          [-4.058865500925465, 5.042133800000001, 12.945102974807526],
          [-4.123541455835809, 4.384043400000001, 12.247964887758233],
          [-4.935174582822345, 5.248007000000001, 13.273579065948546],
          [-3.5204933327218533, 4.697252200000001, 13.66146040761507],
          [-5.415343964721194, 7.099986000000001, 10.98647692071666],
          [-5.409436530453212, 6.487645200000001, 10.247405715649364],
          [-5.729104332590486, 7.9569112, 10.69119788133989],
          [-5.994883670398698, 6.766541800000001, 11.674878941512201],
          [-3.2557154343277603, 9.2396596, 10.070937177323604],
          [-3.3265774931126524, 8.8472688, 9.197330551948552],
          [-2.5208434052696322, 9.8563994, 10.086662096580355],
          [-4.068348472336956, 9.706833399999999, 10.278855554162867],
          [-0.4257597328571543, 8.4170466, 11.400566461144432],
          [-0.00038700894319632084, 7.939315200000001, 10.685956241587641],
          [0.10892458063709753, 8.347542400000002, 12.19554849023573],
          [-0.5189749371993546, 9.341716400000001, 11.157703819290168],
          [-1.0960801834233305, 6.157720200000001, 13.561869252322314],
          [-0.16944825082497283, 6.330161000000001, 13.33298431647405],
          [-1.2204435293149567, 5.218973600000001, 13.771534842412326],
          [-0.2678923768085544, 8.119674200000002, 15.621833674956687],
          [0.9688189356827439, 8.027295200000001, 15.153580523755657],
          [1.1268239408925966, 7.508213200000001, 14.398784399431614],
          [2.0327738241267497, 8.679227000000003, 15.754621882013694],
          [2.89562013755335, 8.590367200000003, 15.419156937869674],
          [1.7854711503132839, 9.460489400000002, 16.85711344323701],
          [2.4829646767664855, 9.935581400000002, 17.25023642465578],
          [0.5426833092315091, 9.546709800000002, 17.37603577870979],
          [0.3932810094324939, 10.038518000000002, 18.151798462042837],
          [-0.5149969609760436, 8.899177000000002, 16.748786221690505],
          [-1.381129150755354, 8.986277200000002, 17.079009526082274],
          [-2.262625953895583, 6.214907200000001, 16.259566511480475],
          [-1.7909280342561744, 4.944476000000002, 16.432540623304735],
          [-1.246580978344844, 4.572320600000001, 15.777335654273447],
          [-2.094652215952071, 4.204564200000002, 17.5367793977788],
          [-1.7743306258384415, 3.3353218000000013, 17.627634486817804],
          [-2.8828822301129913, 4.767636200000002, 18.51521881819886],
          [-3.109257911675553, 4.267030000000002, 19.264773302770656],
          [-3.3392076019901658, 6.0530240000000015, 18.3981555303986],
          [-3.8448395840203813, 6.429578400000002, 19.081315911441894],
          [-3.0465353862929567, 6.792935800000002, 17.264214130661784],
          [-3.3679522685241556, 7.661298400000002, 17.17161182837203],
          [-4.609929746882774, 7.907642400000001, 14.639899828035128],
          [-5.447375721498209, 7.616428600000001, 15.254918892299164],
      ]
      assert mol.radii == [
          1.35,
          1.4,
          1.4,
          0.75,
          0.68,
          0.68,
          0.68,
          0.68,
          0.68,
          0.68,
          0.23,
          0.23,
          0.23,
          0.68,
          0.23,
          0.23,
          0.23,
          0.68,
          0.23,
          0.23,
          0.23,
          0.68,
          0.23,
          0.23,
          0.23,
          0.68,
          0.23,
          0.23,
          0.68,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.23,
          0.68,
          0.68,
      ]
      assert mol.natoms == 52


if __name__== "__main__":
  # test_modules1()
  cell = test_cell2mol()
  test_check_cell(cell)
  test_check_cellrefmoleclist (cell)