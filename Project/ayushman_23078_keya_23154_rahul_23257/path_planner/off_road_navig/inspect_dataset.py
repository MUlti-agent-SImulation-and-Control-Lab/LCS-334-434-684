# inspect_dataset.py
import sys
sys.path.insert(0, '.')
from lidar.tartanground import TartanAirDataset
 
ds = TartanAirDataset('tartanair_data')  # <-- change to your path
ds.print_summary()
