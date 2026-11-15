"""The per-cell offset between a raw ChromBPNet MinGap and the calibrated value the paper reports.

The calibrated value is `raw - DELTA[cell line]`; `aggregate_control_campaign.py` and `paper_numbers.py` read the table.
"""
# The committed per-cell MinGap-contrast offset from random DNA (fig3_main_calibration).
DELTA = {"K562": 2.9068624041974545, "HepG2": -2.9068624041974545}
