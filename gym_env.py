import gym
import numpy as np
from gym import spaces
from xpc import XPlaneConnect


class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        self.xpc = XPlaneConnect()
        try:
            self.xpc.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # начальное состояние
        self.roll = -1.6692969799041748
        self.pitch = 10.215349197387695
        self.yaw = 254.08590698242188
        self.latitude = 55.98773193359375
        self.longitude = 37.517459869384766
        self.elevation = 1510.2215576171875
        self.airspeed = 104.39952087402344
        self.rpm = 2517.006591796875
        self.gear_state = 0.0
        self.flap_position = 1.0

    def reset(self):
        # начальное состояние
        self.roll = -1.6692969799041748
        self.pitch = 10.215349197387695
        self.yaw = 254.08590698242188
        self.latitude = 55.98773193359375
        self.longitude = 37.517459869384766
        self.elevation = 1510.2215576171875
        self.airspeed = 104.39952087402344
        self.rpm = 2517.006591796875
        self.gear_state = 0.0
        self.flap_position = 1.0

        datarefs = []
        values = []
        self.xpc.sendDREFs(datarefs, values)

    def state(self):
        return np.array([self.roll,self.pitch, self.yaw, self.latitude, self.longitude, self.elevation,  self.airspeed,
                         self.rpm, self.gear_state, self.flap_position])

    def step(self, action):
        datarefs = []
        values_set = []
        self.xpc.sendDREFs(datarefs, values_set)

        # делаем степ
        '''
        sim/flightmodel/controls/elv_trim высота тангаж pitch
        sim/flightmodel/controls/ail_trim крен roll
        sim/flightmodel/controls/rud_trim поворот yaw
        sim/flightmodel/engine/ENGN_thro_override изменяем силу тяги (0.0 максимум -2 выключение) (-2; 0)
        
        '''
        target_lat = 55.97421646118164
        target_lon = 37.4338493347168
        target_elevation = 193.03797912597656
        reward = -np.linalg.norm(np.array([self.latitude, self.longitude, self.elevation]) - np.array(
            [target_lat, target_lon, target_elevation]))
        # Loc: (55.97421646118164,) - широта (37.4338493347168,) - долгота (193.03797912597656,) - высота target loc
        # получаем датарефы
        dg = ['sim/flightmodel/position/phi', 'sim/flightmodel/position/theta', 'sim/flightmodel/position/psi', 'sim/flightmodel/position/latitude',
              'sim/flightmodel/position/longitude', 'sim/flightmodel/position/elevation',
              'sim/cockpit2/gauges/indicators/airspeed_kts_pilot', 'sim/cockpit2/engine/indicators/engine_speed_rpm',
              'sim/cockpit2/controls/flap_handle_request_ratio', 'sim/cockpit2/controls/gear_handle_down']
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
	    sim/cockpit2/controls/gear_handle_down Gear handle position. 0 is up. 1 is down. тормоз
	    sim/flightmodel/failures/onground_any 1 если на земле 0 если в воздухе
        '''

        """
        (55.975649, 37.443439, 193.021088) - изначальная точка, в которую появлется самолет когда разбивается
        для того чтобы проверить разбился ли самолет, нужно сравнить широту, долготу, скорость самолета с -21
        """

        # обновляем данные
        self.roll = datarefs_get[0]
        self.pitch = datarefs_get[1]
        self.yaw = datarefs_get[2]
        self.latitude = datarefs_get[3]
        self.longitude = datarefs_get[4]
        self.elevation = datarefs_get[5]
        self.airspeed = datarefs_get[6]
        self.rpm = datarefs_get[7]
        self.flap_position = datarefs_get[8]
        self.gear_state = datarefs_get[9]

        done = False
        if self.latitude == target_lat and self.longitude == target_lon and self.elevation == target_elevation:
            done = True

        return np.array([self.angle_of_attack, self.airspeed, self.pitch_angle, self.bank_angle, self.vertical_speed,
                         self.rpm, self.gear_state, self.flap_position, self.latitude, self.longitude,
                         self.elevation]), reward, done, {}


env = Env()
