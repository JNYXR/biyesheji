Detailed Per Layer Profile
                                                               Bandwidth   time
#    Name                                                MFLOPs  (MB/s)    (ms)
===============================================================================
0    11_conv/Relu                                         173.4   303.2   8.537
1    12_conv/Relu                                        3699.4   664.1  83.121
2    pool1/MaxPool                                          3.2   830.5   7.375
3    21_conv/Relu                                        1849.7   418.7  33.255
4    22_conv/Relu                                        3699.4   473.3  58.830
5    pool2/MaxPool                                          1.6   922.8   3.319
6    31_conv/Relu                                        1849.7   172.7  43.184
7    32_conv/Relu                                        3699.4   181.3  82.242
8    33_conv/Relu                                        3699.4   180.3  82.682
9    pool3/MaxPool                                          0.8   931.1   1.645
10   41_conv/Relu                                        1849.7   137.3  41.531
11   42_conv/Relu                                        3699.4   168.8  67.512
12   43_conv/Relu                                        3699.4   169.6  67.223
13   pool4/MaxPool                                          0.4   922.5   0.831
14   51_conv/Relu                                         924.8   310.0  20.102
15   52_conv/Relu                                         924.8   310.7  20.059
16   53_conv/Relu                                         924.8   307.8  20.245
17   globalmeanpool2d                                       0.2   600.8   0.320
18   dense/bias_add                                         0.0    72.7   0.040
-------------------------------------------------------------------------------
                                   Total inference time                  642.05
-------------------------------------------------------------------------------
