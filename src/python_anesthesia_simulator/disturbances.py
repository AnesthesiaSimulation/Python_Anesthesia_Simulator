import numpy as np


def compute_disturbances(
        time: float,
        dist_profil: str = 'realistic',
        start_step: float = 600,
        end_step: float = 1200,
        start_intub_time: float = 3 * 60,
) -> list:
    """
    Give the value of the distubance profile for a given time.

    Parameters
    ----------
    time : float
        Time: in seconds.
    dist_profil : str, optional
        disturbance profile, can be: 'realistic', 'realistic2', 'liver_transplantation', 'simple', 'step' or "null". The default is 'realistic'.
    start_step : float, optional
        start time of the step distuebance (seconds). The default is 600s.
    end_step : float, optional
        End time of the step distuebance (seconds). The default is 1200s.

    Returns
    -------
    list
        dist_bis, dist_map, dist_co: respectively the additive disturbance to add to the BIS, MAP and CO signals.

    """
    if dist_profil == 'realistic':
        # As proposed in M. M. R. F. Struys, T. De Smet, S. Greenwald, A. R. Absalom, S. Bingé, and E. P. Mortier,
        # “Performance Evaluation of Two Published Closed-loop Control Systems Using Bispectral Index Monitoring:
        #  A Simulation Study,”
        # Anesthesiology, vol. 100, no. 3, pp. 640–647, Mar. 2004, doi: 10.1097/00000542-200403000-00026.

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [9.9,   0,  0, 0],
                                  [10,   20, 10, 0.6],
                                  [12,   20, 10, 0.6],
                                  [13,    0,  0, 0],
                                  [19.9,  0,  0, 0],
                                  [20.2, 20, 10, 0.5],
                                  [21,   20, 10, 0.5],
                                  [21.5,  0,  0, 0],
                                  [26,  -20, -10, -0.8],
                                  [27,   20, 10, 0.9],
                                  [28,   10,  7, 0.2],
                                  [36,   10,  7, 0.2],
                                  [37,   30, 15, 0.8],
                                  [37.5, 30, 15, 0.8],
                                  [38,   10,  5, 0.2],
                                  [41,   10,  5, 0.2],
                                  [41.5, 30, 10, 0.5],
                                  [42,   30, 10, 0.5],
                                  [43,   10,  5, 0.2],
                                  [47,   10,  5, 0.2],
                                  [47.5, 30, 10, 0.9],
                                  [50,   30,  8, 0.9],
                                  [51,   10,  5, 0.2],
                                  [56,   10,  5, 0.2],
                                  [56.5,  0,  0, 0]])

    elif dist_profil == 'realistic2':
        # As proposed in Ionescu, Clara M., et al. "An open source patient simulator for design and evaluation of computer
        # based multiple drug dosing control for anesthetic and hemodynamic variables." IEEE Access 9 (2021): 8680-8694.
        # doi: 10.1109/ACCESS.2021.3049880

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [9.9,   0,  0, 0],
                                  [10,   20, 10, 0.5],
                                  [15,   20, 10, 0.5],
                                  [15.1,  0,  0, 0],
                                  [19.9,  0,  0, 0],
                                  [20,   20, 10, 0.5],
                                  [25,   20, 10, 0.5],
                                  [25.1,  0,  0, 0],
                                  [26.9, -20, -10, -0.5],
                                  [27,   20, 10, 0.5],
                                  [32,   20, 10, 0.5],
                                  [32.1,  0,  0, 0],
                                  [41.9,  0,  0, 0],
                                  [42,   20, 10, 0.5],
                                  [44,   20, 10, 0.5],
                                  [44.1,  0,  0, 0],
                                  [50,    0,  0, 0],
                                  [50.1, 20, 10, 0.5],
                                  [55,   20, 10, 0.5],
                                  [55.1,  0,  0, 0],
                                  [75,    0,  0, 0],
                                  [75.1, 20, 10, 0.5],
                                  [95,   20, 10, 0.5],
                                  [95.1,  0,  0, 0],
                                  [100,   0,  0, 0]])

    elif dist_profil == 'liverTransplantation':
        # The events in this major surgery are Intubation, Incision, recipient hepatectomy, donor liver implementation,

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [9.9,   0,  0, 0],    [10,   20, 10, 0.5],    [
                                      13,   20, 10, 0.5],    [13.1,  0,  0, 0],
                                  [16,    0,  0, 0],    [16.1, 15,  8, 0.4],    [
                                      21,   15,  8, 0.4],    [21.1, 20, 10, 0.5],
                                  [27,   20, 10, 0.5],  [27.1,  0,  0,   0],    [
                                      29,    0,  0,   0],    [29.1,  5,  2, 0.1],
                                  [37,    5,  2, 0.1],  [37.1,  0,  0,   0],    [
                                      39,    0,  0,   0],    [39.1,  5,  2, 0.1],
                                  [46,    5,  2, 0.1],  [46.1,  0,  0,   0],    [
                                      51,    0,  0,   0],    [51.1, 10,  5, 0.2],
                                  [56,   10,  5, 0.2],  [56.1,  0,  0,   0],    [
                                      65,    0,  0,   0],    [65.1, 10,  5, 0.2],
                                  [69,   10,  5, 0.2],  [69.1,  0,  0,   0],    [
                                      78,    0,  0,   0],    [78.1, 10,  5, 0.2],
                                  [82,   10,  5, 0.2],  [82.1,  0,  0,   0],    [
                                      88,    0,  0,   0],    [90,    5,  2, 0.1],
                                  [114,   5,  2, 0.1],  [114.1, 0,  0,   0],    [
                                      116,   0,  0,   0],    [121.5, 0,  0,   0],
                                  [123.5, 5,  2, 0.1],  [125.5, 0,  0,   0],    [
                                      130.5, 0,  0,   0],    [132.5, 5,  2, 0.1],
                                  [134.5, 0,  0, 0],    [141,   0,  0,   0],    [
                                      141.1, 10,  5, 0.2],    [145,  10,  5, 0.2],
                                  [145.1, 0,  0, 0],    [150,   0,  0,   0],    [
                                      151,  10,  5, 0.2],    [155,  10,  5, 0.2],
                                  [156,   5,  2, 0.1],  [157,  10,  5, 0.2],    [
                                      161,  10,  5, 0.2],    [162,   0,  0,   0],
                                  [165,   0,  0, 0],    [166,   5,  2, 0.1],    [
                                      169,   5,  2, 0.1],    [169.1, 10,  5, 0.2],
                                  [171,  10,  5, 0.2],  [172,   0,  0,   0],    [
                                      173,   0,  0,   0],    [173.5, 10,  5, 0.2],
                                  [174,   0,  0, 0],    [181,   0,  0,   0],    [
                                      181.1, 15,  8, 0.4],    [183.5, 15,  8, 0.4],
                                  [183.6, 0,  0, 0],    [186,   0,  0,   0],    [
                                      186.1, 10,  5, 0.2],    [189,  10,  5, 0.2],
                                  [189.1, 0,  0, 0],    [190,   0,  0,   0],    [
                                      190.1, 5,  2, 0.1],    [193,  5,   2, 0.1],
                                  [193.1, 0,  0, 0],    [196,   0,  0,   0],    [
                                      198,   8,  4, 0.1],    [204,  8,   4, 0.1],
                                  [206,   0,  0, 0],    [208,   0,  0,   0],    [
                                      210,  10,  5, 0.2],    [222, 10,   5, 0.2],
                                  [224,   0,  0, 0],    [226,   5,  2, 0.1],    [
                                      227,  12,  6,  0.3],   [232, 12,   6, 0.3],
                                  [234,   0,  0, 0],    [237,   0,  0,   0],    [
                                      238,   8,  4, 0.1],    [251,  8,   4, 0.1],
                                  [252,   0,  0, 0],    [260,   0,  0,   0],    [
                                      263,  15,  8, 0.4],    [270, 15,   8, 0.4],
                                  [273,   5,  2, 0.1],  [338,   5,  2, 0.1],    [341,  0,  0,    0],    [350,  0,   0,   0]])

    elif dist_profil == 'simple':
        # As in G. A. Dumont, A. Martinez, and J. M. Ansermino,
        # “Robust control of depth of anesthesia,”
        # International Journal of Adaptive Control and Signal Processing,
        # vol. 23, no. 5, pp. 435–454, 2009, doi: 10.1002/acs.1087.

        Disturb_point = np.array([[0,     0,  0, 0],  # time, BIS signal, MAP, CO signals
                                  [19.9,  0,  0, 0],
                                  [20,   20,  5, 0.3],
                                  [23,   20, 10, 0.6],
                                  [24,   15, 10, 0.6],
                                  [26, 12.5,  6, 0.4],
                                  [30, 10.5,  4, 0.3],
                                  [37,   10,  4, 0.3],
                                  [40,    4,  2, 0.1],
                                  [45,  0.5, 0.1, 0.01],
                                  [50,    0,  0,   0]])
    elif dist_profil == 'step':
        Disturb_point = np.array([[0,     0,  0,   0],  # time, BIS signal, MAP, CO signals
                                  [start_step / 60 - 0.01,   0,  0,   0],
                                  [start_step / 60,    10,  5, 0.3],
                                  [end_step / 60 - 0.01,   10,  5, 0.3],
                                  [end_step / 60,  0,  0,   0],
                                  [30,    0,  0,   0]])
    elif dist_profil == 'intubation':
        # Disturbance profile (A) as proposed in M. M. R. F. Struys, T. De Smet, S. Greenwald, A. R. Absalom, S. Bingé, and E. P. Mortier,
        # “Performance Evaluation of Two Published Closed-loop Control Systems Using Bispectral Index Monitoring:
        #  A Simulation Study,”
        # Anesthesiology, vol. 100, no. 3, pp. 640–647, Mar. 2004, doi: 10.1097/00000542-200403000-00026.
        Disturb_point = np.array([ # time, BIS signal, MAP, CO signals
            [0, 0, 0, 0],
            [start_step/60 - 0.01, 0, 0, 0],
            [start_step/60, 20, 10, 0.5],
            [end_step/60 - 0.01, 20, 10, 0.5],
            [end_step/60, 0, 0, 0],
            [30, 0, 0, 0]
        ])
    elif dist_profil == 'null':
        return [0, 0, 0]
    elif dist_profil == 'VitalDB':
        p = [
            -8.60412601e-11,
            1.53817088e-07,
            -9.35659716e-05,
            1.14438483e-02,
            8.07370704e+00,
        ]
        dist_bis = np.polyval(p, time - start_intub_time - 4 * 60)
        dist_map = 0
        dist_co = 0
        if time < start_intub_time:
            dist_bis = 0
        elif dist_bis < 0:
            dist_bis = 0
        return [dist_bis, dist_map, dist_co]
    else:
        raise ValueError(
            'dist_profil should be: realistic, realistic2, liverTransplantation, simple, step, VitalDB or null')

    dist_bis = np.interp(time / 60, Disturb_point[:, 0], Disturb_point[:, 1])
    dist_map = np.interp(time / 60, Disturb_point[:, 0], Disturb_point[:, 2])
    dist_co = np.interp(time / 60, Disturb_point[:, 0], Disturb_point[:, 3])

    return [dist_bis, dist_map, dist_co]
