import gym
import numpy as np
from gym import spaces
from xpc import XPlaneConnect


class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        self.xpc = XPlaneConnect()
        try:
            self.xpc.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # начальное состояние
        self.angle_of_attack = 0.0
        self.airspeed = 0.0
        self.pitch_angle = 0.0
        self.bank_angle = 0.0
        self.vertical_speed = 0.0
        self.rpm = 0.0
        self.gear_state = 0.0
        self.flap_position = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.elevation = 0.0

    def reset(self):
        # начальное состояние
        self.angle_of_attack = 0.0
        self.airspeed = 0.0
        self.pitch_angle = 0.0
        self.bank_angle = 0.0
        self.vertical_speed = 0.0
        self.rpm = 0.0
        self.gear_state = 0.0
        self.flap_position = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.elevation = 0.0

        datarefs = []
        values = []
        self.xpc.sendDREFs(datarefs, values)

    def state(self):
        return np.array([self.angle_of_attack, self.airspeed, self.pitch_angle, self.bank_angle, self.vertical_speed,
                         self.rpm, self.gear_state, self.flap_position, self.latitude, self.longitude, self.elevation])

    def step(self, action):
        datarefs = []
        values_set = []
        self.xpc.sendDREFs(datarefs, values_set)

        # делаем степ
        '''
        sim/flightmodel/controls/elv_trim
        
        '''
        target_lat = 1.0
        target_lon = 1.0
        target_elevation = 1.0
        reward = -np.linalg.norm(np.array([self.latitude, self.longitude, self.elevation]) - np.array([target_lat, target_lon, target_elevation]))

        # получаем датарефы
        dg = []
        datarefs_get = self.xpc.getDREFs(dg)

        # обновляем данные
        self.angle_of_attack = datarefs_get[0]
        self.airspeed = datarefs_get[1]
        self.pitch_angle = datarefs_get[2]
        self.bank_angle = datarefs_get[3]
        self.vertical_speed = datarefs_get[4]
        self.rpm = datarefs_get[5]
        self.gear_state = datarefs_get[6]
        self.flap_position = datarefs_get[7]

        done = False
        if self.latitude == target_lat and self.longitude == target_lon and self.elevation == target_elevation:
            done = True

        return np.array([self.angle_of_attack, self.airspeed, self.pitch_angle, self.bank_angle, self.vertical_speed,
                         self.rpm, self.gear_state, self.flap_position, self.latitude, self.longitude,
                         self.elevation]), reward, done, {}

