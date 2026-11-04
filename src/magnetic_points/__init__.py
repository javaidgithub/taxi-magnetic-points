"""Magnetic Points: taxi pickup hotspot recommendation with recursive K-Means.

Companion code for the paper "Integrating AI in Ride-Hailing: A Framework for
Taxi Hotspot Recommendation Using Recursive K-Means Clustering and
Classification" (see README.md for the reference).

Pipeline order::

    preprocessing -> clustering -> evaluation -> prediction -> visualization
"""

__version__ = "0.1.0"

DAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}
