"""
Where the Sun is in the sky at a given moment, as seen from a given place.

Low-precision solar ephemeris after Meeus, "Astronomical Algorithms", ch. 25
(the same formulation the NOAA solar calculator uses). Its error is about
0.01 degrees — at a typical eclipse focal length that is well under a pixel,
and far smaller than the error of a camera clock that is off by ten seconds.

Everything the eclipse-sequence composite needs lives here:
  * apparent altitude / azimuth (with atmospheric refraction, which lifts a Sun
    near the horizon by half a degree — a whole solar diameter),
  * the Sun's apparent angular radius on that date,
  * the ecliptic, for the optional sky overlay.

No third-party astronomy package is required.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

# The Sun's radius seen from 1 AU (IAU nominal value), in degrees.
SUN_RADIUS_AT_1AU_DEG = 959.63 / 3600.0


def _normalize_deg(angle: float) -> float:
    return angle % 360.0


def to_utc(moment: datetime, utc_offset_hours: float = 0.0) -> datetime:
    """
    Converts a camera timestamp to UTC.

    A naive datetime (what EXIF DateTimeOriginal gives) is interpreted as local
    time at `utc_offset_hours`; an aware one already knows its offset.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone(timedelta(hours=float(utc_offset_hours))))
    return moment.astimezone(timezone.utc)


def julian_day(moment_utc: datetime) -> float:
    """Julian Day of a UTC datetime (naive values are taken as UTC)."""
    if moment_utc.tzinfo is not None:
        moment_utc = moment_utc.astimezone(timezone.utc).replace(tzinfo=None)
    # 2000-01-01 12:00 UTC is JD 2451545.0; timedelta arithmetic keeps full precision.
    delta = moment_utc - datetime(2000, 1, 1, 12, 0, 0)
    return 2451545.0 + delta.total_seconds() / 86400.0


def _solar_coordinates(jd: float) -> Tuple[float, float, float, float]:
    """
    Apparent right ascension, declination (degrees), distance (AU) and the true
    obliquity of the ecliptic (degrees) for a Julian Day.
    """
    t = (jd - 2451545.0) / 36525.0

    l0 = _normalize_deg(280.46646 + t * (36000.76983 + t * 0.0003032))
    m = _normalize_deg(357.52911 + t * (35999.05029 - t * 0.0001537))
    e = 0.016708634 - t * (0.000042037 + t * 0.0000001267)
    m_rad = math.radians(m)

    c = ((1.914602 - t * (0.004817 + t * 0.000014)) * math.sin(m_rad)
         + (0.019993 - t * 0.000101) * math.sin(2 * m_rad)
         + 0.000289 * math.sin(3 * m_rad))
    true_long = l0 + c
    anomaly = m + c
    distance = 1.000001018 * (1 - e * e) / (1 + e * math.cos(math.radians(anomaly)))

    omega = 125.04 - 1934.136 * t
    apparent_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))

    lam = math.radians(apparent_long)
    eps_rad = math.radians(eps)
    ra = math.degrees(math.atan2(math.cos(eps_rad) * math.sin(lam), math.cos(lam)))
    dec = math.degrees(math.asin(math.sin(eps_rad) * math.sin(lam)))
    return _normalize_deg(ra), dec, distance, eps


def greenwich_sidereal_time_deg(jd: float) -> float:
    """Greenwich mean sidereal time in degrees (Meeus 12.4)."""
    t = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + t * t * (0.000387933 - t / 38710000.0))
    return _normalize_deg(gmst)


def equatorial_to_horizontal(ra_deg: float, dec_deg: float, jd: float,
                             latitude_deg: float, longitude_deg: float) -> Tuple[float, float]:
    """
    (azimuth, true altitude) in degrees. Azimuth is measured from north through
    east; longitude is positive east of Greenwich.
    """
    lst = greenwich_sidereal_time_deg(jd) + longitude_deg
    hour_angle = math.radians(_normalize_deg(lst - ra_deg))
    lat = math.radians(latitude_deg)
    dec = math.radians(dec_deg)

    sin_alt = (math.sin(lat) * math.sin(dec)
               + math.cos(lat) * math.cos(dec) * math.cos(hour_angle))
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(-math.cos(dec) * math.sin(hour_angle),
                    math.sin(dec) * math.cos(lat)
                    - math.cos(dec) * math.cos(hour_angle) * math.sin(lat))
    return _normalize_deg(math.degrees(az)), math.degrees(alt)


def refraction_deg(true_altitude_deg: float, pressure_hpa: float = 1010.0,
                   temperature_c: float = 10.0) -> float:
    """
    Atmospheric refraction for a TRUE altitude (Saemundsson), in degrees.

    It is ~0.57 deg at the horizon, 0.09 deg at 10 deg and negligible above
    45 deg. Below about -2 deg the formula diverges, so it is held constant —
    such a Sun is below any real horizon anyway.
    """
    h = max(true_altitude_deg, -1.9)
    r_arcmin = 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11)))
    r_arcmin *= (pressure_hpa / 1010.0) * (283.0 / (273.0 + temperature_c))
    return max(0.0, r_arcmin / 60.0)


def sun_position(moment: datetime, latitude_deg: float, longitude_deg: float,
                 utc_offset_hours: float = 0.0, refraction: bool = True) -> Tuple[float, float]:
    """
    Apparent (azimuth, altitude) of the Sun's centre in degrees.

    `moment` may be naive local time (then `utc_offset_hours` applies) or an
    aware datetime.
    """
    jd = julian_day(to_utc(moment, utc_offset_hours))
    ra, dec, _dist, _eps = _solar_coordinates(jd)
    az, alt = equatorial_to_horizontal(ra, dec, jd, latitude_deg, longitude_deg)
    if refraction:
        alt += refraction_deg(alt)
    return az, alt


def sun_angular_radius_deg(moment: datetime, utc_offset_hours: float = 0.0) -> float:
    """The Sun's apparent angular radius on that date (0.262..0.271 deg)."""
    jd = julian_day(to_utc(moment, utc_offset_hours))
    _ra, _dec, dist, _eps = _solar_coordinates(jd)
    return SUN_RADIUS_AT_1AU_DEG / dist


def sun_equatorial(moment: datetime, utc_offset_hours: float = 0.0) -> Tuple[float, float]:
    """Apparent (right ascension, declination) of the Sun in degrees."""
    jd = julian_day(to_utc(moment, utc_offset_hours))
    ra, dec, _dist, _eps = _solar_coordinates(jd)
    return ra, dec


def sun_path(start: datetime, end: datetime, step_minutes: float,
             latitude_deg: float, longitude_deg: float,
             utc_offset_hours: float = 0.0) -> List[Tuple[datetime, float, float]]:
    """(time, azimuth, apparent altitude) samples of the Sun's daily path."""
    if end < start:
        start, end = end, start
    step = timedelta(minutes=max(0.05, float(step_minutes)))
    samples = []
    moment = start
    # Guard against an absurd request producing millions of samples.
    for _ in range(20000):
        if moment > end:
            break
        az, alt = sun_position(moment, latitude_deg, longitude_deg, utc_offset_hours)
        samples.append((moment, az, alt))
        moment += step
    return samples


def ecliptic_horizontal(moment: datetime, latitude_deg: float, longitude_deg: float,
                        utc_offset_hours: float = 0.0,
                        step_deg: float = 1.0) -> List[Tuple[float, float, float]]:
    """
    The ecliptic great circle at one instant, as (ecliptic longitude, azimuth,
    apparent altitude) samples. The Sun always lies on it.
    """
    jd = julian_day(to_utc(moment, utc_offset_hours))
    _ra, _dec, _dist, eps = _solar_coordinates(jd)
    eps_rad = math.radians(eps)
    points = []
    n = max(8, int(round(360.0 / max(0.1, step_deg))))
    for i in range(n + 1):
        lam = math.radians(360.0 * i / n)
        ra = math.degrees(math.atan2(math.cos(eps_rad) * math.sin(lam), math.cos(lam)))
        dec = math.degrees(math.asin(math.sin(eps_rad) * math.sin(lam)))
        az, alt = equatorial_to_horizontal(_normalize_deg(ra), dec, jd,
                                           latitude_deg, longitude_deg)
        # Always applied (the formula is held constant deep below the horizon),
        # so the drawn line has no kink where refraction would switch off.
        alt += refraction_deg(alt)
        points.append((math.degrees(lam), az, alt))
    return points
