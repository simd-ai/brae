# G4 tank mesh (2,483,933 cells) decomposePar statistics

decomposePar -constant on the G4 tank-refill case (mesh only), stats by decomp_stats.py. scotch is not deterministic from run to run (np=8: 28444 cut faces here, 27223 in another run).

### scotch_2

## decomposition: g4
| rank | cells | interface faces | neighbours | neighbour:faces |
|---|---|---|---|---|
| 0 | 1240637 | 8757 | 1 | 1:8757 |
| 1 | 1243296 | 8757 | 1 | 0:8757 |

np=2: cells 2483933 (min 1240637 / max 1243296, imbalance 0.1%); cut faces 8757; interface faces/rank min 8757 max 8757 (max/cells 0.7%); neighbours/rank min 1 max 1

### scotch_4

## decomposition: g4
| rank | cells | interface faces | neighbours | neighbour:faces |
|---|---|---|---|---|
| 0 | 616729 | 7149 | 3 | 1:4788 2:1514 3:847 |
| 1 | 621428 | 11369 | 3 | 0:4788 2:4557 3:2024 |
| 2 | 625490 | 11823 | 3 | 0:1514 1:4557 3:5752 |
| 3 | 620286 | 8623 | 3 | 0:847 1:2024 2:5752 |

np=4: cells 2483933 (min 616729 / max 625490, imbalance 0.7%); cut faces 19482; interface faces/rank min 7149 max 11823 (max/cells 1.9%); neighbours/rank min 3 max 3

### scotch_8

## decomposition: g4
| rank | cells | interface faces | neighbours | neighbour:faces |
|---|---|---|---|---|
| 0 | 309674 | 5342 | 5 | 1:1778 2:1865 3:58 6:161 7:1480 |
| 1 | 309856 | 5751 | 5 | 0:1778 3:633 4:1 5:2187 7:1152 |
| 2 | 310615 | 5878 | 5 | 0:1865 3:1985 4:327 6:1607 7:94 |
| 3 | 309455 | 6468 | 6 | 0:58 1:633 2:1985 4:2272 5:1501 7:19 |
| 4 | 310561 | 7416 | 6 | 1:1 2:327 3:2272 5:2279 6:1830 7:707 |
| 5 | 309448 | 8279 | 5 | 1:2187 3:1501 4:2279 6:870 7:1442 |
| 6 | 312503 | 7443 | 5 | 0:161 2:1607 4:1830 5:870 7:2975 |
| 7 | 311821 | 7869 | 7 | 0:1480 1:1152 2:94 3:19 4:707 5:1442 6:2975 |

np=8: cells 2483933 (min 309448 / max 312503, imbalance 0.6%); cut faces 27223; interface faces/rank min 5342 max 8279 (max/cells 2.7%); neighbours/rank min 5 max 7

### simple_8

## decomposition: g4
| rank | cells | interface faces | neighbours | neighbour:faces |
|---|---|---|---|---|
| 0 | 310523 | 7019 | 1 | 4:7019 |
| 1 | 309261 | 6658 | 1 | 5:6658 |
| 2 | 311182 | 6963 | 1 | 6:6963 |
| 3 | 311001 | 7070 | 1 | 7:7070 |
| 4 | 310808 | 17621 | 3 | 0:7019 5:5191 6:5411 |
| 5 | 311375 | 17022 | 3 | 1:6658 4:5191 7:5173 |
| 6 | 309454 | 17769 | 3 | 2:6963 4:5411 7:5395 |
| 7 | 310329 | 17638 | 3 | 3:7070 5:5173 6:5395 |

np=8: cells 2483933 (min 309261 / max 311375, imbalance 0.3%); cut faces 48880; interface faces/rank min 6658 max 17769 (max/cells 5.7%); neighbours/rank min 1 max 3

### hier_8

## decomposition: g4
| rank | cells | interface faces | neighbours | neighbour:faces |
|---|---|---|---|---|
| 0 | 310491 | 7087 | 2 | 1:7018 4:69 |
| 1 | 310491 | 18295 | 4 | 0:7018 3:5699 4:1 5:5577 |
| 2 | 310491 | 7756 | 2 | 3:6956 6:800 |
| 3 | 310492 | 18321 | 4 | 1:5699 2:6956 6:7 7:5659 |
| 4 | 310492 | 8204 | 4 | 0:69 1:1 5:6658 6:1476 |
| 5 | 310492 | 17419 | 3 | 1:5577 4:6658 7:5184 |
| 6 | 310492 | 9353 | 4 | 2:800 3:7 4:1476 7:7070 |
| 7 | 310492 | 17913 | 3 | 3:5659 5:5184 6:7070 |

np=8: cells 2483933 (min 310491 / max 310492, imbalance 0.0%); cut faces 52174; interface faces/rank min 7087 max 18321 (max/cells 5.9%); neighbours/rank min 2 max 4
