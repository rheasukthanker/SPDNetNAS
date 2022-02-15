from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]


radar_sparsemax = Genotype(normal=[('Skip_normal', 0), ('Skip_normal', 1),
                                   ('BiMap_0_normal', 0),
                                   ('BiMap_1_normal', 2)],
                           normal_concat=range(2, 4),
                           reduce=[('BiMap_2_reduced', 1),
                                   ('BiMap_2_reduced', 0),
                                   ('BiMap_1_reduced', 1),
                                   ('BiMap_2_reduced', 0)],
                           reduce_concat=range(2, 4))
PRIMITIVES_SPDNet_v0 = [
    'none_normal', 'WeightedPooling_normal', 'BiMap_0_normal',
    'BiMap_1_normal', 'BiMap_2_normal', 'Skip_normal', 'none_reduced',
    'AvgPooling2_reduced', 'MaxPooling_reduced', 'BiMap_1_reduced',
    'BiMap_2_reduced', 'Skip_reduced'
]
#'BiMap_0_reduced',
PRIMITIVES_SPDNet_v1 = [
    'none_normal', 'AvgPooling_1_normal', 'DiMap_1_normal', 'DiMap_2_normal',
    'BiMap_1_normal', 'BiMap_2_normal', 'Skip_1_normal', 'none_reduced',
    'AvgPooling_2_reduced', 'MaxPooling_reduced', 'BiMap_0_reduced',
    'BiMap_1_reduced', 'BiMap_2_reduced', 'Skip_2_reduced'
]

hdm05_sparsemax = Genotype(normal=[('Skip_normal', 0), ('Skip_normal', 1),
                                      ('BiMap_2_normal', 0),
                                      ('BiMap_2_normal', 2)],
                              normal_concat=range(2, 4),
                              reduce=[('BiMap_1_reduced', 0),
                                      ('BiMap_1_reduced', 1),
                                      ('BiMap_2_reduced', 0),
                                      ('BiMap_1_reduced', 1)],
                              reduce_concat=range(2, 4))







