from datetime import datetime
from typing import Any, List, Sequence
import numpy as np
from gymnasium import Env
from sinergym.utils.constants import YEAR


class MyRuleBasedController(object):

    def __init__(self, env: Env) -> None:
        """Rule-based controller for 2-room environment with two heating setpoints (TRVs).
        Controls Living Room and Bedroom heating setpoints based on room temperatures,
        outdoor conditions, and time of day.

        Args:
            env (Env): Simulation environment
        """
        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        # Comfort temperature ranges based on season
        self.comfort_range_winter = (20.0, 23.5)  # Winter comfort range
        self.comfort_range_summer = (23.0, 26.0)  # Summer comfort range

        # Setpoint adjustment parameters
        self.temp_adjustment = 1.0  # Temperature adjustment step
        self.night_setback = 2.0    # Night setback amount

        # Schedule parameters
        self.night_start = 22  # 10 PM
        self.night_end = 6     # 6 AM

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select heating setpoint actions for both Living Room and Bedroom TRVs.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen (LR_heating_setpoint, BR_heating_setpoint).
        """
        obs_dict = dict(zip(self.observation_variables, observation))

        # Extract time information with bounds checking
        raw_day = obs_dict.get('day_of_month', 1)
        raw_month = obs_dict.get('month', 1)

        day = max(1, min(31, int(raw_day)))    # Ensure day is between 1-31
        month = max(1, min(12, int(raw_month)))  # Ensure month is between 1-12
        hour = int(obs_dict.get('hour', 0))
        year = int(obs_dict.get('year', YEAR))

        # Determine season
        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)
        current_dt = datetime(year, month, day)

        is_summer = summer_start_date <= current_dt <= summer_final_date
        comfort_range = self.comfort_range_summer if is_summer else self.comfort_range_winter

        # Extract temperature readings
        outdoor_temp = obs_dict['outdoor_temperature']
        lr_temp = obs_dict['air_temp_lr']
        br_temp = obs_dict['air_temp_br']

        # Determine if it's night time or weekend
        is_night = hour >= self.night_start or hour < self.night_end
        is_weekend = current_dt.weekday() >= 5  # Saturday = 5, Sunday = 6

        # Calculate base setpoints for each room
        lr_setpoint = self._calculate_room_setpoint(lr_temp, outdoor_temp, comfort_range, is_night, is_weekend)
        br_setpoint = self._calculate_room_setpoint(br_temp, outdoor_temp, comfort_range, is_night, is_weekend)

        # Clip setpoints to action space bounds
        action_space = self.env.get_wrapper_attr('action_space')
        lr_setpoint = np.clip(
            lr_setpoint,
            action_space.low[0],
            action_space.high[0])
        br_setpoint = np.clip(
            br_setpoint,
            action_space.low[1],
            action_space.high[1])

        return np.array([lr_setpoint, br_setpoint], dtype=np.float32)

    def _calculate_room_setpoint(
            self,
            room_temp: float,
            outdoor_temp: float,
            comfort_range: tuple,
            is_night: bool,
            is_weekend: bool) -> float:
        """Calculate heating setpoint for a specific room based on conditions.

        Args:
            room_temp (float): Current room temperature
            outdoor_temp (float): Current outdoor temperature
            comfort_range (tuple): (min_comfort, max_comfort) temperatures
            is_night (bool): Whether it's night time
            is_weekend (bool): Whether it's weekend

        Returns:
            float: Calculated heating setpoint
        """
        min_comfort, max_comfort = comfort_range

        # Base setpoint (middle of comfort range)
        base_setpoint = (min_comfort + max_comfort) / 2.0

        # Apply night setback
        if is_night or is_weekend:
            base_setpoint -= self.night_setback

        # Adjust based on room temperature relative to comfort range
        if room_temp < min_comfort:
            # Room is too cold, increase setpoint
            temp_deficit = min_comfort - room_temp
            setpoint = base_setpoint + \
                min(temp_deficit * 0.5, 3.0)  # Cap increase at 3°C
        elif room_temp > max_comfort:
            # Room is too warm, decrease setpoint
            temp_excess = room_temp - max_comfort
            setpoint = base_setpoint - \
                min(temp_excess * 0.5, 2.0)  # Cap decrease at 2°C
        else:
            # Room is in comfort range
            setpoint = base_setpoint

        # Adjust based on outdoor temperature (weather compensation)
        if outdoor_temp < 0:
            # Very cold outside, increase setpoint
            setpoint += 1.0
        elif outdoor_temp < 10:
            # Cold outside, slight increase
            setpoint += 0.5
        elif outdoor_temp > 20:
            # Warm outside, decrease setpoint
            setpoint -= 0.5

        # Ensure minimum heating setpoint (prevent freezing)
        setpoint = max(setpoint, 15.0)

        return setpoint
