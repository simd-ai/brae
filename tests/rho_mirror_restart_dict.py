#!/usr/bin/env python3
# Point a controlDict at the latest written time and give it three more steps, for the restart arms.
# endTime is ABSOLUTE in OpenFOAM, so `startTime + 3` is what makes the run three iterations long.
import re
import sys

c, n = sys.argv[1], int(sys.argv[2])
s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %d;' % (n + 3), s)
open(c, 'w').write(s)
