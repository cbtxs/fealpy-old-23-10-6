import numpy as np


from fealpy.mesh import EdgeMesh

node = np.array([
      [1,   3005.57544,    1503.7854,          50.],
      [2,   3005.57544,    3003.7854,          50.],
      [3,   2818.07544,    3003.7854,         100.],
      [4,   2818.07544,    1503.7854,         100.],
      [5,   2443.07544,    1503.7854,           0.],
      [6,   2818.07544,    1503.7854,           0.],
      [7,   3193.07544,    1503.7854,           0.],
      [8,   3193.07544,   3.78534746,           0.],
      [9,   3005.57544,   3.78534746,          50.],
     [10,   3193.07544,    3003.7854,           0.],
     [11,   2818.07544,    3003.7854,           0.],
     [12,   2443.07544,    3003.7854,           0.],
     [13,   2630.57544,    1503.7854,         150.],
     [14,   2630.57544,    3003.7854,         150.],
     [15,   2443.07544,    3003.7854,         200.],
     [16,   2443.07544,    1503.7854,         200.],
     [17,   2068.07544,    1503.7854,           0.],
     [18,   2068.07544,    1503.7854,         300.],
     [19,   2068.07544,   3.78534746,         300.],
     [20,   2443.07544,   3.78534746,           0.],
     [21,   2443.07544,   3.78534746,         200.],`
     [22,   2630.57544,   3.78534746,         150.],
     [23,   2818.07544,   3.78534746,         100.],
     [24,   2818.07544,   3.78534746,           0.],
     [25,   2068.07544,    3003.7854,         300.],
     [26,   2255.57544,    3003.7854,         250.],
     [27,   2255.57544,    1503.7854,         250.],
     [28,   2255.57544,   3.78534746,         250.],
     [29,   1693.07544,    1503.7854,           0.],
     [30,   1880.57544,   3.78534746,         350.],
     [31,   1880.57544,    1503.7854,         350.],
     [32,   1693.07544,    1503.7854,         400.],
     [33,   1693.07544,   3.78534746,         400.],
     [34,   1505.57544,   3.78534746,         350.],
     [35,   1505.57544,    1503.7854,         350.],
     [36,   1318.07544,    1503.7854,         300.],
     [37,   1318.07544,    3003.7854,         300.],
     [38,     943.0755,    3003.7854,           0.],
     [39,   1318.07544,    3003.7854,           0.],
     [40,   1693.07544,    3003.7854,           0.],
     [41,   2068.07544,    3003.7854,           0.],
     [42,   2068.07544,   3.78534746,           0.],
     [43,   1880.57544,    3003.7854,         350.],
     [44,   1693.07544,    3003.7854,         400.],
     [45,   1693.07544,   3.78534746,           0.],
     [46,   1318.07544,   3.78534746,         300.],
     [47,   1318.07544,   3.78534746,           0.],
     [48,     943.0755,   3.78534746,           0.],
     [49,     568.0755,   3.78534746,         100.],
     [50,     755.5755,   3.78534746,         150.],
     [51,     943.0755,   3.78534746,         200.],
     [52,     193.0755,   3.78534746,           0.],
     [53,     193.0755,    1503.7854,           0.],
     [54,     943.0755,    1503.7854,           0.],
     [55,   1130.57544,    1503.7854,         250.],
     [56,   1130.57544,    3003.7854,         250.],
     [57,     943.0755,    3003.7854,         200.],
     [58,     943.0755,    1503.7854,         200.],
     [59,   1505.57544,    3003.7854,         350.],
     [60,   1318.07544,    1503.7854,           0.],
     [61,     568.0755,    3003.7854,         100.],
     [62,     755.5755,    3003.7854,         150.],
     [63,     193.0755,    3003.7854,           0.],
     [64,     380.5755,    1503.7854,          50.],
     [65,     568.0755,    1503.7854,         100.],
     [66,     755.5755,    1503.7854,         150.],
     [67,     380.5755,   3.78534746,          50.],
     [68,   1130.57544,   3.78534746,         250.],
     [69,     380.5755,    3003.7854,          50.]], dtype=np.float64)

cell0 = np.array([
    [1, 1, 2],
    [3, 4, 3],
    [4, 5, 4],
    [7, 8, 7],
     [9, 9, 1],
    [12,  3, 11],
    [13, 12,  3],
    [16,  7, 10],
    [19, 13, 14],
    [21, 16, 15],
    [22, 16,  5],
    [24, 18, 17],
    [25, 19, 18],
    [26, 20, 19],
    [27, 21, 20],
    [31,  5, 18],
    [32, 4, 6],
    [34, 23, 24],
    [35, 20, 23],
    [38, 12, 25],
    [39, 15, 12],
    [40, 23,  4],
    [41, 22, 13],
    [43, 27, 26],
    [44, 28, 27],
    [46, 21, 16],
    [49, 18, 29],
    [54, 30, 31],
    [56, 33, 32],
    [58, 34, 35],
    [60, 36, 37],
    [61, 38, 37],
    [67, 19, 42],
    [69, 31, 43],
    [71, 40, 44],
    [72, 37, 40],
    [73, 37, 39],
    [76, 18, 25],
    [77, 29, 32],
    [78, 36, 29],
    [79, 19, 45],
    [81, 45, 33],
    [82, 46, 45],
    [83, 46, 47],
    [85, 49, 48],
    [88, 51, 48],
    [90, 52, 53],
    [92, 54, 36],
    [94, 55, 56],
    [96, 58, 57],
    [97, 58, 54],
     [99, 35, 59],
    [102, 46, 36],
    [103, 36, 60],
    [105, 61, 38],
    [108, 57, 38],
    [110, 53, 63],
    [116, 25, 40],
    [117, 25, 41],
    [119, 32, 44],
    [122, 49, 65],
    [124, 67, 64],
    [125, 50, 66],
    [127, 51, 58],
    [128, 48, 46],
    [131, 65, 54],
    [132, 68, 55],
    [134, 65, 61],
    [135, 66, 62],
    [137, 64, 69]], dtype=np.int_)
