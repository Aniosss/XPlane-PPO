import time

import gym
import numpy as np
from gym import spaces

import xpc
from xpc import XPlaneConnect

START_G_FORCE = 1.0
START_ROLL = -1.6692969799041748
START_PITCH = 10.215349197387695
START_YAW = 254.08590698242188
START_LATITUDE = 55.98773193359375
START_LONGITUDE = 37.517459869384766
START_ELEVATION = 1510.2215576171875
START_AIRSPEED = 104.39952087402344
START_RPM = 1363.704345703125
START_FLAP_POSITION = 0.0
START_PARKING_BRAKE = 0.0

TARGET_LAT = 55.97421646118164
TARGET_LON = 37.4338493347168
TARGET_ELEVATION = 193.03797912597656

EPS = 1e-4

DATAREFS_STATE = ['sim/flightmodel/position/phi', 'sim/flightmodel/position/theta', 'sim/flightmodel/position/psi',
              'sim/flightmodel/position/latitude', 'sim/flightmodel/position/longitude',
              'sim/flightmodel/position/elevation',
              'sim/cockpit2/gauges/indicators/airspeed_kts_pilot', 'sim/cockpit2/engine/indicators/engine_speed_rpm',
              'sim/flightmodel/controls/parkbrake']

START_VALUES = [START_ROLL, START_PITCH, START_YAW, START_LATITUDE, START_LONGITUDE, START_ELEVATION, START_AIRSPEED,
                START_RPM, START_PARKING_BRAKE]


class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
        self.action_space = spaces.Box(low=-2, high=1, shape=(5,))

        self.xpc = XPlaneConnect()

        # начальное состояние
        self.g_force = START_G_FORCE
        self.roll = START_ROLL
        self.pitch = START_PITCH
        self.yaw = START_YAW
        self.latitude = START_LATITUDE
        self.longitude = START_LONGITUDE
        self.elevation = START_ELEVATION
        self.airspeed = START_AIRSPEED
        self.rpm = START_RPM
        self.parking_brake = START_PARKING_BRAKE

    def reset(self):
        #       Lat             Lon              Alt              Pitch        Roll        Yaw       Gear
        posi = [START_LATITUDE, START_LONGITUDE, START_ELEVATION, START_PITCH, START_ROLL, START_YAW, 0]
        xpc.XPlaneConnect().sendPOSI(posi)
        xpc.XPlaneConnect().sendDREFs(DATAREFS_STATE, START_VALUES)

        # начальное состояние
        self.roll = START_ROLL
        self.pitch = START_PITCH
        self.yaw = START_YAW
        self.latitude = START_LATITUDE
        self.longitude = START_LONGITUDE
        self.elevation = START_ELEVATION
        self.airspeed = START_AIRSPEED
        self.rpm = START_RPM
        self.parking_brake = START_PARKING_BRAKE

        datarefs = DATAREFS_STATE
        values = START_VALUES
        self.xpc.sendDREFs(datarefs, values)
        return values

    def state(self):
        return np.array([self.roll, self.pitch, self.yaw, self.latitude, self.longitude, self.elevation, self.airspeed,
                         self.rpm, self.parking_brake])

    def step(self, action):
        datarefs = ['sim/flightmodel/controls/elv_trim', 'sim/flightmodel/controls/ail_trim',
                    'sim/flightmodel/controls/rud_trim', 'sim/flightmodel/engine/ENGN_thro_override',
                    'sim/flightmodel/controls/parkbrake']
        values_set = action
        self.xpc.sendDREFs(datarefs, values_set)

        # делаем степ
        '''
        sim/flightmodel/controls/elv_trim тангаж pitch [-1..1]
        sim/flightmodel/controls/ail_trim крен roll [-1..1]
        sim/flightmodel/controls/rud_trim поворот yaw [-1..1]
        sim/flightmodel/engine/ENGN_thro_override изменяем силу тяги (0.0 максимум -2 выключение)(-1..1) - 1 -> (-2; 0)
        sim/flightmodel/controls/parkbrake [0..1] (-1..1) -> max(0, x)  тормоз parking brake 1 = max
        '''

        target_lat = TARGET_LAT
        target_lon = TARGET_LON
        target_elevation = TARGET_ELEVATION
        reward = -np.linalg.norm(np.array([self.latitude, self.longitude, self.elevation]) - np.array(
            [target_lat, target_lon, target_elevation])) - (abs(self.g_force) - 1) * 100

        # Loc: (55.97421646118164,) - широта (37.4338493347168,) - долгота (193.03797912597656,) - высота target loc
        # получаем датарефы

        dg = ['sim/flightmodel/position/phi', 'sim/flightmodel/position/theta', 'sim/flightmodel/position/psi',
              'sim/flightmodel/position/latitude', 'sim/flightmodel/position/longitude',
              'sim/flightmodel/position/elevation',
              'sim/cockpit2/gauges/indicators/airspeed_kts_pilot', 'sim/cockpit2/engine/indicators/engine_speed_rpm',
              'sim/flightmodel/controls/parkbrake', 'sim/flightmodel/forces/g_nrml']
        datarefs_get = self.xpc.getDREFs(dg)

        '''
        sim/flightmodel/position/phi угол крена в градусах
        sim/flightmodel/position/theta тангаж
        sim/flightmodel/position/psi угол поворота относительно оси Z рысканье
        sim/flightmodel/position/latitude широта
        sim/flightmodel/position/longitude долгота
        sim/flightmodel/position/elevation высота метры
        sim/cockpit2/engine/indicators/engine_speed_rpm rpm
        sim/cockpit2/gauges/indicators/airspeed_kts_pilot Indicated airspeed in knots, pilot. Writeable with override_IAS скорость самолета в узлах
        sim/cockpit2/controls/flap_handle_request_ratio The flap HANDLE location, in ratio, where 0.0 is handle fully retracted, and 1.0 is handle fully extended. закрылки
	    sim/flightmodel/failures/onground_any 1 если на земле 0 если в воздухе
	    sim/flightmodel/controls/parkbrake положение тормоза
	    sim/flightmodel/forces/g_nrml g_force 
        '''

        """
        (55.975649, 37.443439, 193.021088) - изначальная точка, в которую появлется самолет когда разбивается
        (55.97564697265625,), (37.44343948364258,), (193.02459716796875,), (-21.0,)
        для того чтобы проверить разбился ли самолет, нужно сравнить широту, долготу, скорость самолета с -21
        """

        # обновляем данные
        self.roll = datarefs_get[0][0]
        self.pitch = datarefs_get[1][0]
        self.yaw = datarefs_get[2][0]
        self.latitude = datarefs_get[3][0]
        self.longitude = datarefs_get[4][0]
        self.elevation = datarefs_get[5][0]
        self.airspeed = datarefs_get[6][0]
        self.rpm = datarefs_get[7][0]
        self.parking_brake = datarefs_get[8][0]
        self.g_force = datarefs_get[9][0]
        done = False
        if (target_lat - EPS <= self.latitude <= target_lat + EPS) and (
                target_lon - EPS <= self.longitude <= target_lon + EPS) and (
                target_elevation - EPS <= self.elevation <= target_elevation + EPS):
            done = True

        return np.array([self.roll, self.pitch, self.yaw, self.latitude, self.longitude, self.elevation, self.airspeed,
                         self.rpm, self.parking_brake]), reward, done, {}

