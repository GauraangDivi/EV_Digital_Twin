# tata_nexon_ev_digital_twin_prototype.py
"""
TATA NEXON EV DIGITAL TWIN PROTOTYPE
Real-time Digital Twin with OEM Dataset Calibration Support
Max battery temperature: 60°C (warning threshold). Thermal runaway if >= 120°C.

Changes in this version:
- Fixed Streamlit slider error when min == max by guarding brake slider.
- Charging cannot be started if vehicle speed > 0 or accelerator > 0.
- When charging is active, accelerator is locked to 0 (no propulsion).
- Added additional analytical graphs in the Analytics tab to satisfy OEM requirements:
    * Cumulative energy (kWh) timeline (computed from twin power history)
    * Recovered energy timeline (regen energy)
    * SOC vs Speed 2D density (heatmap)
    * Battery Temperature rolling mean & distribution (box + rolling line)
    * Scatter plots: Power vs Speed, SOC vs Temperature
- Uses st.rerun() when needed (no experimental_rerun usage).
- Powertrain tab: RPM now strictly follows vehicle speed:
    * RPM == 0 at speed == 0
    * RPM increases with speed and is maximal at configured top_speed_kmh
- Powertrain tab now shows *motor shaft power* clipped to max_power_kw.
  Battery tab shows *battery power* (net kW at the pack, including losses and regen),
  so the power values in different tabs are intentionally different but physically consistent.
- Reduced RPM update latency by increasing UI refresh frequency and capping dt to 0.2s.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from enum import Enum
from typing import Optional

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="TATA NEXON EV DIGITAL TWIN PROTOTYPE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #F8F9FA; }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    h1, h2, h3 { color: #0D47A1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================== ENUMS ====================


class DrivingMode(Enum):
    """Tata Nexon EV Driving Modes"""
    CITY = "City Mode"
    ECO = "Eco Mode"
    SPORT = "Sport Mode"


class ChargingProtocol(Enum):
    """Charging Protocols"""
    CCS2 = "CCS2 (DC Fast - 50kW)"
    AC_CHARGING = "AC Charging (7.2kW)"
    HOME_CHARGING = "Home Charging (3.3kW)"


# ==================== BATTERY MANAGEMENT SYSTEM ====================


class TataNexonBMS:
    """Tata Nexon EV Battery Management System"""

    def __init__(self, config):
        self.config = config
        self.soc = 85.0
        self.soh = 100.0
        self.voltage = 325.0
        self.current = 0.0
        self.temperature = 25.0
        self.power = 0.0
        self.energy_consumed = 0.0
        self.cycle_count = 0

        self.coolant_temp = 25.0
        self.cooling_active = False
        self.heating_active = False

        self.kf_estimate = self.soc
        self.kf_error_covariance = 1.0
        self.process_noise = 0.01
        self.measurement_noise = 0.1

        # Default OCV map (will be replaced if calibrated from dataset)
        self.soc_points = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        self.ocv_points = np.array([260, 285, 300, 310, 320, 325, 330, 340, 350, 365, 380])

    def _soc_to_ocv(self, soc):
        """OCV lookup table (calibrated if dataset is provided)"""
        return np.interp(soc, self.soc_points, self.ocv_points)

    def estimate_soc_coulomb_counting(self, current_a, dt_hours):
        capacity_ah = self.config["capacity_kwh"] * 1000.0 / self.config["nominal_voltage"]
        delta_soc = (current_a * dt_hours / capacity_ah) * 100.0
        # current sign convention: positive = discharge, negative = charging
        self.soc = np.clip(self.soc - delta_soc, 0.0, 100.0)
        return self.soc

    def estimate_soc_kalman_filter(self, measured_voltage):
        prediction = self.kf_estimate
        prediction_error = self.kf_error_covariance + self.process_noise

        ocv_expected = self._soc_to_ocv(prediction)
        innovation = measured_voltage - ocv_expected

        kalman_gain = prediction_error / (prediction_error + self.measurement_noise)
        self.kf_estimate = prediction + kalman_gain * innovation
        self.kf_error_covariance = (1 - kalman_gain) * prediction_error

        self.kf_estimate = np.clip(self.kf_estimate, 0.0, 100.0)
        return self.kf_estimate

    def estimate_soh(self):
        capacity_fade = 100.0 * np.exp(-0.00003 * self.cycle_count)
        self.soh = np.clip(capacity_fade, 70.0, 100.0)
        return self.soh

    def advanced_thermal_management(self, ambient_temp, power_dissipation_kw):
        """
        Improved thermal model:
        - includes Joule heating from current
        - includes a fraction of charging/discharging power that converts to heat
        - returns temperature and active cooling/heating flags
        """
        # Joule heating from current (A^2 * R_effective)
        heat_from_current_w = (self.current ** 2) * 0.015  # empirical

        # A fraction of battery electrical power becomes heat (both during discharge and charging)
        heat_from_power_w = abs(power_dissipation_kw) * 1000.0 * 0.02  # 2% -> heat (tunable)

        heat_generation_w = heat_from_current_w + heat_from_power_w

        # Thermal control logic
        if self.temperature >= 120.0:
            # Extremely high temperature => thermal runaway
            self.cooling_active = True
            self.heating_active = False
            heat_dissipation_w = 0.0  # runaway
        elif self.temperature > 35.0:
            self.cooling_active = True
            self.heating_active = False
            cooling_power = 1500.0
            heat_dissipation_w = cooling_power * (self.temperature - ambient_temp) / 10.0
        elif self.temperature < 15.0:
            self.heating_active = True
            self.cooling_active = False
            heating_power = 800.0
            heat_dissipation_w = -heating_power * (ambient_temp - self.temperature) / 10.0
        else:
            self.cooling_active = False
            self.heating_active = False
            heat_dissipation_w = 300.0 * (self.temperature - ambient_temp) / 10.0

        # Simple thermal integration
        thermal_mass_j_per_c = 150000.0
        dt_seconds = 1.0
        temp_change_c = (heat_generation_w - heat_dissipation_w) * dt_seconds / thermal_mass_j_per_c
        self.temperature = self.temperature + temp_change_c

        self.coolant_temp = self.temperature - 5.0 if self.cooling_active else self.temperature
        self.temperature = max(self.temperature, -40.0)
        return self.temperature, self.cooling_active, self.heating_active

    def safety_check(self):
        warnings = []
        errors = []

        if self.temperature >= 120.0:
            errors.append("CRITICAL: Thermal runaway detected")
            return warnings, errors

        if self.voltage > self.config["max_voltage"] * 1.05:
            errors.append("CRITICAL: Overvoltage detected")
        elif self.voltage > self.config["max_voltage"]:
            warnings.append("Warning: High voltage")

        if self.temperature > 60.0:
            errors.append("CRITICAL: Battery overheating")
        elif self.temperature > 50.0:
            warnings.append("Warning: High temperature")

        if self.temperature < 0.0:
            errors.append("CRITICAL: Battery too cold")
        elif self.temperature < 10.0:
            warnings.append("Warning: Low temperature")

        if self.soc < 3.0:
            errors.append("CRITICAL: Battery critically low")
        elif self.soc < 10.0:
            warnings.append("Warning: Low battery")

        return warnings, errors

    def update(self, current_a, ambient_temp, dt_seconds=1.0, power_dissipation_kw: float = 0.0):
        dt_hours = dt_seconds / 3600.0

        self.current = current_a
        self.voltage = self._soc_to_ocv(self.soc)
        # positive = discharge, negative = charging
        self.power = (self.voltage * self.current) / 1000.0  # kW

        soc_coulomb = self.estimate_soc_coulomb_counting(current_a, dt_hours)
        soc_kalman = self.estimate_soc_kalman_filter(self.voltage)
        self.soc = 0.7 * soc_kalman + 0.3 * soc_coulomb

        if abs(current_a) > 10.0:
            self.cycle_count += dt_hours * 0.00005
        self.estimate_soh()

        self.temperature, cooling_active, heating_active = self.advanced_thermal_management(
            ambient_temp, power_dissipation_kw
        )

        warnings, errors = self.safety_check()
        self.energy_consumed += abs(self.power * dt_hours)

        if self.temperature >= 120.0 and "CRITICAL: Thermal runaway detected" not in errors:
            errors.append("CRITICAL: Thermal runaway detected")

        return {
            "soc": self.soc,
            "soh": self.soh,
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temperature,
            "power": self.power,
            "cooling_active": cooling_active,
            "heating_active": heating_active,
            "coolant_temp": self.coolant_temp,
            "warnings": warnings,
            "errors": errors,
        }


# ==================== BRAKING SYSTEM ====================


class TataNexonBrakingSystem:
    """Advanced Braking System with Regenerative Braking"""

    def __init__(self):
        self.brake_pressure = 0.0
        self.regen_level = 2
        self.mechanical_brake_active = False
        self.regen_brake_active = False
        self.abs_active = False
        self.energy_recovered_kwh = 0.0
        self.brake_temp_front = 25.0
        self.brake_temp_rear = 25.0

    def calculate_braking_force(self, brake_pedal_pct, vehicle_speed_kmh):
        self.brake_pressure = brake_pedal_pct

        if vehicle_speed_kmh > 20.0:
            regen_effectiveness = 1.0
        elif vehicle_speed_kmh > 10.0:
            regen_effectiveness = 0.7
        elif vehicle_speed_kmh > 5.0:
            regen_effectiveness = 0.3
        else:
            regen_effectiveness = 0.0

        regen_multipliers = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9}
        regen_factor = regen_multipliers.get(self.regen_level, 0.6) * regen_effectiveness

        regen_force_pct = brake_pedal_pct * regen_factor
        mechanical_force_pct = brake_pedal_pct * (1.0 - regen_factor)

        self.regen_brake_active = regen_force_pct > 5.0
        self.mechanical_brake_active = mechanical_force_pct > 5.0
        self.abs_active = brake_pedal_pct > 80.0 and vehicle_speed_kmh > 30.0

        return regen_force_pct, mechanical_force_pct

    def calculate_regen_power(self, vehicle_speed_kmh, regen_force_pct):
        if not self.regen_brake_active:
            return 0.0

        max_regen_power = 35.0  # kW
        speed_factor = min(vehicle_speed_kmh / 100.0, 1.0)
        regen_power_kw = -max_regen_power * (regen_force_pct / 100.0) * speed_factor * 0.85

        return regen_power_kw

    def update_brake_temperature(self, mechanical_force_pct, vehicle_speed_kmh, ambient_temp):
        if self.mechanical_brake_active:
            heat_gen_front = mechanical_force_pct * vehicle_speed_kmh * 0.02
            heat_gen_rear = mechanical_force_pct * vehicle_speed_kmh * 0.015
        else:
            heat_gen_front = 0.0
            heat_gen_rear = 0.0

        cooling_rate = 0.5 + (vehicle_speed_kmh * 0.01)

        self.brake_temp_front += (heat_gen_front - cooling_rate * (self.brake_temp_front - ambient_temp)) * 0.1
        self.brake_temp_rear += (heat_gen_rear - cooling_rate * (self.brake_temp_rear - ambient_temp)) * 0.1

        self.brake_temp_front = max(ambient_temp, self.brake_temp_front)
        self.brake_temp_rear = max(ambient_temp, self.brake_temp_rear)

    def update(self, brake_pedal_pct, vehicle_speed_kmh, ambient_temp, dt_seconds=1.0, charging_active=False):
        regen_force, mech_force = self.calculate_braking_force(brake_pedal_pct, vehicle_speed_kmh)
        regen_power = self.calculate_regen_power(vehicle_speed_kmh, regen_force)

        if charging_active:
            regen_power = 0.0
            self.regen_brake_active = False

        if regen_power < 0.0:
            self.energy_recovered_kwh += abs(regen_power) * (dt_seconds / 3600.0)

        self.update_brake_temperature(mech_force, vehicle_speed_kmh, ambient_temp)

        return {
            "brake_pressure": self.brake_pressure,
            "regen_active": self.regen_brake_active,
            "mechanical_active": self.mechanical_brake_active,
            "abs_active": self.abs_active,
            "regen_power_kw": regen_power,
            "regen_level": self.regen_level,
            "energy_recovered": self.energy_recovered_kwh,
            "brake_temp_front": self.brake_temp_front,
            "brake_temp_rear": self.brake_temp_rear,
        }


# ==================== POWERTRAIN ====================


class TataNexonPowertrain:
    """Tata Nexon EV Powertrain"""

    def __init__(self, config, driving_mode=DrivingMode.CITY):
        self.config = config
        self.driving_mode = driving_mode
        self.motor_rpm = 0.0
        self.motor_torque = 0.0
        self.motor_efficiency = 0.92
        self.inverter_efficiency = 0.96

    def set_driving_mode(self, mode):
        self.driving_mode = mode

    def calculate_motor_speed(self, vehicle_speed_kmh, accelerator_pct: float = 0.0):
        """
        Calculate motor RPM from vehicle speed.
        Strict mapping: at 0 km/h -> 0 RPM; increases linearly with speed using wheel diameter and gear ratio.
        """
        wheel_diameter_m = 0.665
        gear_ratio = 10.5
        # wheel_rpm = (speed_m_per_min) / (circumference)
        wheel_rpm = (vehicle_speed_kmh * 1000.0 / 60.0) / (np.pi * wheel_diameter_m)
        motor_rpm = wheel_rpm * gear_ratio

        # No artificial RPM floor: exactly 0 at 0 speed and max at top_speed_kmh.
        self.motor_rpm = motor_rpm
        return motor_rpm

    def calculate_torque_demand(self, accelerator_pct, vehicle_speed_kmh):
        max_torque = self.config["max_torque_nm"]

        if self.driving_mode == DrivingMode.ECO:
            base_torque = (accelerator_pct / 100.0) * max_torque * 0.80
        elif self.driving_mode == DrivingMode.CITY:
            base_torque = (accelerator_pct / 100.0) * max_torque * 0.90
        elif self.driving_mode == DrivingMode.SPORT:
            base_torque = (accelerator_pct / 100.0) * max_torque * 1.0
        else:
            base_torque = (accelerator_pct / 100.0) * max_torque

        if vehicle_speed_kmh > 100.0:
            derating_factor = 1.0 - (vehicle_speed_kmh - 100.0) / 120.0
            base_torque *= max(0.4, derating_factor)

        self.motor_torque = base_torque
        return self.motor_torque

    def calculate_power_from_torque_and_rpm(self, motor_torque_nm, motor_rpm):
        """
        Compute motor shaft (mechanical) power from torque and rpm (kW),
        apply max power limit, and derive approximate battery power (kW).

        Returns:
            mech_kw (float): motor shaft power (kW), limited to config["max_power_kw"]
            batt_kw (float): corresponding battery power (kW), incl. losses
                             (positive = discharge, negative = charging)
            limited_torque_nm (float): torque after applying power limit (for UI)
        """
        if motor_torque_nm is None:
            motor_torque_nm = 0.0
        if motor_rpm is None or motor_rpm == 0.0 or motor_torque_nm == 0.0:
            return 0.0, 0.0, motor_torque_nm

        # mechanical power (W) = torque (Nm) * angular_speed (rad/s)
        power_w = motor_torque_nm * motor_rpm * 2.0 * np.pi / 60.0
        mech_kw_raw = power_w / 1000.0  # mechanical kW

        max_mech_kw = self.config.get("max_power_kw", 105.0)

        abs_raw = abs(mech_kw_raw)
        if abs_raw > max_mech_kw:
            scale = max_mech_kw / abs_raw
            mech_kw = mech_kw_raw * scale  # ±max_mech_kw
            limited_torque = motor_torque_nm * scale
        else:
            mech_kw = mech_kw_raw
            limited_torque = motor_torque_nm

        # battery power estimation
        if mech_kw > 0.0:
            # traction: battery supplies more than shaft due to losses
            batt_kw = mech_kw / (self.motor_efficiency * self.inverter_efficiency)
        elif mech_kw < 0.0:
            # regen: battery receives less than shaft magnitude due to losses
            batt_kw = mech_kw * (self.motor_efficiency * self.inverter_efficiency)
        else:
            batt_kw = 0.0

        return mech_kw, batt_kw, limited_torque

    def update(self, accelerator_pct, vehicle_speed_kmh):
        """
        Backwards-compatible update: the main twin now uses the
        lower-level methods directly, but this is kept for any
        other callers.
        """
        self.motor_torque = self.calculate_torque_demand(accelerator_pct, vehicle_speed_kmh)
        self.motor_rpm = self.calculate_motor_speed(vehicle_speed_kmh, accelerator_pct)
        mech_kw, batt_kw, limited_torque = self.calculate_power_from_torque_and_rpm(
            self.motor_torque, self.motor_rpm
        )

        return {
            "motor_rpm": self.motor_rpm,
            "motor_torque": limited_torque,
            "power_kw": mech_kw,  # shaft power
            "driving_mode": self.driving_mode.value,
        }


# ==================== VEHICLE DYNAMICS ====================


class TataNexonDynamics:
    """Vehicle dynamics"""

    def __init__(self, config):
        self.config = config
        self.speed_kmh = 0.0
        self.acceleration_mps2 = 0.0
        self.distance_km = 0.0

    def update(self, motor_torque_nm, brake_force_pct, dt_seconds=1.0, allow_acceleration=True):
        mass_kg = self.config.get("weight_kg", 1400.0)
        wheel_radius_m = 0.3325
        gear_ratio = 10.5
        air_drag_coeff = 0.33
        frontal_area_m2 = 2.4
        rolling_resistance_coeff = 0.012

        wheel_force_n = (motor_torque_nm * gear_ratio) / wheel_radius_m
        brake_force_n = brake_force_pct * 50.0
        drag_force_n = 0.5 * 1.225 * air_drag_coeff * frontal_area_m2 * (self.speed_kmh / 3.6) ** 2
        rolling_resistance_n = rolling_resistance_coeff * mass_kg * 9.81

        if not allow_acceleration:
            net_force_n = -brake_force_n - drag_force_n - rolling_resistance_n
        else:
            net_force_n = wheel_force_n - brake_force_n - drag_force_n - rolling_resistance_n

        self.acceleration_mps2 = net_force_n / mass_kg

        delta_speed_mps = self.acceleration_mps2 * dt_seconds
        self.speed_kmh += delta_speed_mps * 3.6
        self.speed_kmh = max(0.0, min(self.speed_kmh, self.config.get("top_speed_kmh", 140.0)))

        self.distance_km += (self.speed_kmh / 3600.0) * dt_seconds

        return {
            "speed_kmh": self.speed_kmh,
            "acceleration_mps2": self.acceleration_mps2,
            "distance_km": self.distance_km,
        }


# ==================== CHARGING SYSTEM ====================


class TataNexonChargingSystem:
    """Charging System"""

    def __init__(self):
        self.charging_active = False
        self.charging_protocol = None
        self.charging_power_kw = 0.0
        self.charging_current_a = 0.0
        self.charge_time_remaining_min = 0.0

    def initiate_charging(self, protocol, target_soc, current_soc, battery_capacity_kwh):
        self.charging_protocol = protocol
        self.charging_active = True

        # detect protocol string loosely (caller sends the enum .value)
        if "50" in protocol:
            self.charging_power_kw = 50.0
            self.charging_current_a = 125
        elif "7.2" in protocol:
            self.charging_power_kw = 7.2
            self.charging_current_a = 32
        else:
            self.charging_power_kw = 3.3
            self.charging_current_a = 16

        energy_needed = max(0.0, (target_soc - current_soc) / 100.0 * battery_capacity_kwh)
        if self.charging_power_kw > 0.0:
            self.charge_time_remaining_min = (energy_needed / self.charging_power_kw) * 60.0
        else:
            self.charge_time_remaining_min = 0.0

        return {
            "status": "Charging initiated",
            "protocol": self.charging_protocol,
            "power_kw": self.charging_power_kw,
            "time_remaining_min": self.charge_time_remaining_min,
        }

    def update_charging(self, current_soc, target_soc, battery_temp, dt_seconds=1.0):
        if not self.charging_active:
            return {"charging_active": False}

        if battery_temp > 40.0:
            temp_factor = 0.7
        elif battery_temp < 15.0:
            temp_factor = 0.8
        else:
            temp_factor = 1.0

        if current_soc > 80.0:
            soc_factor = 1.0 - ((current_soc - 80.0) / 20.0) * 0.6
            soc_factor = max(0.2, soc_factor)
        else:
            soc_factor = 1.0

        actual_power = self.charging_power_kw * temp_factor * soc_factor

        self.charge_time_remaining_min = max(0.0, self.charge_time_remaining_min - dt_seconds / 60.0)

        if current_soc >= target_soc:
            self.stop_charging()
            return {"charging_active": False, "status": "Charging complete"}

        return {
            "charging_active": True,
            "actual_power_kw": actual_power,
            "time_remaining_min": self.charge_time_remaining_min,
        }

    def stop_charging(self):
        self.charging_active = False
        self.charging_power_kw = 0.0
        self.charging_current_a = 0.0


# ==================== DIGITAL TWIN ====================


class TataNexonEVDigitalTwin:
    """Complete Tata Nexon EV Digital Twin"""

    def __init__(self, vehicle_id="TN-NEXON-001", dataset: Optional[pd.DataFrame] = None):
        self.vehicle_id = vehicle_id

        self.config = {
            "model": "Tata Nexon EV (40.5 kWh)",
            "capacity_kwh": 40.5,
            "usable_capacity_kwh": 38.8,
            "nominal_voltage": 325.0,
            "max_voltage": 380.0,
            "min_voltage": 260.0,
            "max_discharge_current": 300.0,
            "max_power_kw": 105.0,
            "max_torque_nm": 250.0,
            "range_km": 312.0,
            "weight_kg": 1400.0,
            "top_speed_kmh": 140.0,
            "acceleration_0_100": 9.9,
        }

        self.bms = TataNexonBMS(self.config)
        self.powertrain = TataNexonPowertrain(self.config)
        self.braking = TataNexonBrakingSystem()
        self.dynamics = TataNexonDynamics(self.config)
        self.charging = TataNexonChargingSystem()

        self.ambient_temperature = 30.0
        self.accelerator_position = 0.0
        self.brake_position = 0.0
        self.driving_mode = DrivingMode.CITY

        self.dataset = dataset

        self.shutdown = False

        self.history = {
            "time": [],
            "soc": [],
            "soh": [],
            "speed": [],
            "power": [],          # battery power (kW)
            "temperature": [],
            "energy_recovered": [],
        }

    def update(self, dt_seconds=0.2):
        if self.shutdown:
            self.charging.stop_charging()
            bms_state = self.bms.update(0.0, self.ambient_temperature, dt_seconds, power_dissipation_kw=0.0)
            dynamics_state = {
                "speed_kmh": self.dynamics.speed_kmh,
                "acceleration_mps2": 0.0,
                "distance_km": self.dynamics.distance_km,
            }
            braking_state = {
                "brake_pressure": self.brake_position,
                "regen_active": False,
                "mechanical_active": False,
                "abs_active": False,
                "regen_power_kw": 0.0,
                "regen_level": self.braking.regen_level,
                "energy_recovered": self.braking.energy_recovered_kwh,
                "brake_temp_front": self.braking.brake_temp_front,
                "brake_temp_rear": self.braking.brake_temp_rear,
            }
            self._append_history(bms_state, dynamics_state, braking_state)
            return {
                "bms": bms_state,
                "powertrain": {
                    "motor_rpm": 0,
                    "motor_torque": 0,
                    "power_kw": 0,
                    "driving_mode": self.driving_mode.value,
                },
                "braking": braking_state,
                "dynamics": dynamics_state,
                "estimated_range_km": 0.0,
            }

        # Decide whether propulsion is allowed (no propulsion while charging)
        if self.charging.charging_active:
            accel_input = 0.0
            allow_accel = False
        else:
            accel_input = self.accelerator_position
            allow_accel = True

        # ----- SEQUENCING: braking -> torque (with brake derate) -> dynamics -> rpm/power -> BMS -----

        # 1) braking state (uses current speed)
        braking_state = self.braking.update(
            self.brake_position,
            self.dynamics.speed_kmh,
            self.ambient_temperature,
            dt_seconds,
            charging_active=self.charging.charging_active,
        )

        regen_power = braking_state["regen_power_kw"]  # kW, negative when charging via regen

        # 2) torque demand from accelerator (based on current speed)
        raw_torque_nm = self.powertrain.calculate_torque_demand(accel_input, self.dynamics.speed_kmh)

        # 3) Derate torque as brake increases so that motor power reduces with braking
        # up to 90% torque cut at full brake
        if allow_accel:
            brake_fraction = min(1.0, braking_state["brake_pressure"] / 100.0)
            torque_derate_factor = 1.0 - 0.9 * brake_fraction
            torque_nm = raw_torque_nm * torque_derate_factor
        else:
            torque_nm = 0.0

        # 4) compute brake force used by dynamics (mechanical vs regen scaling)
        brake_force = braking_state["brake_pressure"] * (1 if braking_state["mechanical_active"] else 0.3)

        # 5) update vehicle dynamics using torque (wheel force) and brake force
        dynamics_state = self.dynamics.update(
            torque_nm,
            brake_force,
            dt_seconds,
            allow_acceleration=allow_accel,
        )

        # 6) compute motor rpm from the *updated* speed (strict mapping: 0 -> 0 RPM)
        motor_rpm = self.powertrain.calculate_motor_speed(dynamics_state["speed_kmh"], accel_input)

        # 7) compute mechanical motor power & corresponding battery power
        mech_power_kw, batt_power_kw, limited_torque = self.powertrain.calculate_power_from_torque_and_rpm(
            torque_nm, motor_rpm
        )

        # build powertrain state (used for UI + history)
        powertrain_state = {
            "motor_rpm": motor_rpm,
            "motor_torque": limited_torque,
            "power_kw": mech_power_kw,  # shaft power shown in Powertrain tab
            "driving_mode": self.driving_mode.value,
        }

        # 8) battery current & dissipation: combine traction battery power and regen power
        # batt_power_kw: +ve = battery discharge to motor, -ve = charge due to traction regen
        if (not self.charging.charging_active) and (
            abs(batt_power_kw) > 0.0 or abs(regen_power) > 0.0
        ):
            # total net battery power (kW): traction + regen from braking system
            total_batt_power_kw = batt_power_kw + regen_power
            battery_current = (total_batt_power_kw * 1000.0) / max(self.bms.voltage, 1.0)
            power_dissipation_kw = abs(total_batt_power_kw)
        else:
            total_batt_power_kw = 0.0
            battery_current = 0.0
            power_dissipation_kw = 0.0

        # 9) if charging active, override battery_current using charging model
        if self.charging.charging_active:
            charging_state = self.charging.update_charging(
                self.bms.soc, 100.0, self.bms.temperature, dt_seconds
            )
            if charging_state.get("charging_active", False):
                actual_charging_power_kw = charging_state["actual_power_kw"]
                total_batt_power_kw = -actual_charging_power_kw  # negative = charging
                battery_current = (
                    -actual_charging_power_kw * 1000.0 / max(self.bms.voltage, 1.0)
                )
                power_dissipation_kw = abs(actual_charging_power_kw)
            else:
                self.charging.stop_charging()
                total_batt_power_kw = 0.0
                power_dissipation_kw = 0.0

        # 10) update BMS with the computed battery_current and power dissipation
        bms_state = self.bms.update(
            battery_current,
            self.ambient_temperature,
            dt_seconds,
            power_dissipation_kw=power_dissipation_kw,
        )
        # Note: bms_state["power"] is the net battery power (kW) and is what gets logged in history

        # 11) handle thermal runaway / shutdown
        if "CRITICAL: Thermal runaway detected" in bms_state.get("errors", []):
            self.shutdown = True
            self.charging.stop_charging()
            st.warning(
                "🚨 Thermal runaway detected — twin entering safe shutdown (charging stopped)."
            )

        # 12) append history and compute estimated range
        self._append_history(bms_state, dynamics_state, braking_state)

        remaining_energy = (bms_state["soc"] / 100.0) * self.config["usable_capacity_kwh"]
        avg_consumption = 0.124
        estimated_range = remaining_energy / avg_consumption if avg_consumption > 0.0 else 0.0

        return {
            "bms": bms_state,
            "powertrain": powertrain_state,
            "braking": braking_state,
            "dynamics": dynamics_state,
            "estimated_range_km": estimated_range,
        }

    def _append_history(self, bms_state, dynamics_state, braking_state):
        self.history["time"].append(datetime.now())
        self.history["soc"].append(bms_state["soc"])
        self.history["soh"].append(bms_state["soh"])
        self.history["speed"].append(dynamics_state["speed_kmh"])
        # battery-side power history
        self.history["power"].append(bms_state["power"])
        self.history["temperature"].append(bms_state["temperature"])
        self.history["energy_recovered"].append(braking_state["energy_recovered"])

        max_history = 500
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]


# ==================== STREAMLIT UI ====================


def initialize_session_state():
    """Initialize session state"""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "ev_twin" not in st.session_state:
        st.session_state.ev_twin = TataNexonEVDigitalTwin(dataset=st.session_state.dataset)
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if "current_state" not in st.session_state:
        ev = st.session_state.ev_twin
        remaining_energy = (ev.bms.soc / 100.0) * ev.config["usable_capacity_kwh"]
        st.session_state.current_state = {
            "bms": {
                "soc": ev.bms.soc,
                "soh": ev.bms.soh,
                "voltage": ev.bms.voltage,
                "current": ev.bms.current,
                "temperature": ev.bms.temperature,
                "power": ev.bms.power,
                "cooling_active": False,
                "heating_active": False,
                "coolant_temp": ev.bms.coolant_temp,
                "warnings": [],
                "errors": [],
            },
            "powertrain": {
                "motor_rpm": 0,
                "motor_torque": 0,
                "power_kw": 0,
                "driving_mode": ev.driving_mode.value,
            },
            "braking": {
                "brake_pressure": 0,
                "regen_active": False,
                "mechanical_active": False,
                "abs_active": False,
                "regen_power_kw": 0,
                "regen_level": ev.braking.regen_level,
                "energy_recovered": ev.braking.energy_recovered_kwh,
                "brake_temp_front": ev.braking.brake_temp_front,
                "brake_temp_rear": ev.braking.brake_temp_rear,
            },
            "dynamics": {
                "speed_kmh": ev.dynamics.speed_kmh,
                "acceleration_mps2": 0,
                "distance_km": ev.dynamics.distance_km,
            },
            "estimated_range_km": remaining_energy / 0.124,
        }


@st.fragment(run_every="0.2s")
def render_realtime_kpis():
    """Real-time KPI updates - only this section refreshes"""
    ev = st.session_state.ev_twin

    if st.session_state.simulation_running:
        current_time = time.time()
        dt = current_time - st.session_state.last_update
        # cap dt to 0.2s for faster updates and reduced latency
        state = ev.update(dt_seconds=min(dt, 0.2))
        st.session_state.last_update = current_time
        st.session_state.current_state = state
    else:
        state = st.session_state.current_state

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "🔋 Battery SOC",
            f"{state['bms']['soc']:.1f}%",
            delta=f"SOH: {state['bms']['soh']:.1f}%",
        )

    with col2:
        ev_local = st.session_state.ev_twin
        if ev_local.charging.charging_active:
            power_label = "Charging"
        else:
            power_label = "Regen" if state["bms"]["power"] < 0.0 else "Consuming"

        # Battery-side power (kW)
        st.metric(
            "⚡ Power (Battery)",
            f"{abs(state['bms']['power']):.1f} kW",
            delta=power_label,
        )

    with col3:
        st.metric(
            "🏎️ Speed",
            f"{state['dynamics']['speed_kmh']:.1f} km/h",
            delta=f"{state['dynamics']['acceleration_mps2']:.2f} m/s²",
        )

    with col4:
        temp_status = (
            "Cooling"
            if state["bms"]["cooling_active"]
            else "Heating"
            if state["bms"]["heating_active"]
            else "Normal"
        )
        st.metric(
            "🌡️ Battery Temp",
            f"{state['bms']['temperature']:.1f}°C",
            delta=temp_status,
        )

    with col5:
        st.metric(
            "📍 Range",
            f"{state['estimated_range_km']:.0f} km",
            delta=f"{state['dynamics']['distance_km']:.1f} km driven",
        )

    if state["bms"]["errors"]:
        for error in state["bms"]["errors"]:
            st.error(f"🚨 {error}")

    if state["bms"]["warnings"]:
        for warning in state["bms"]["warnings"]:
            st.warning(f"⚠️ {warning}")


# ---------------- Helper: analytics computations ----------------


def compute_cumulative_energy_from_history(ev: TataNexonEVDigitalTwin):
    """
    Compute cumulative absolute energy (kWh) from power history using timestamps.
    Power here is battery power from BMS (kW, signed).
    Returns times (datetime list) and cumulative_energy (kWh list) aligned with times.
    """
    times = ev.history["time"]
    powers = ev.history["power"]
    if len(times) < 2:
        # trivial
        return times, [0.0] * len(times)

    # convert to seconds
    t_seconds = np.array([t.timestamp() for t in times], dtype=float)
    p = np.array(powers, dtype=float)  # kW (signed)
    cumulative = [0.0]
    for i in range(1, len(t_seconds)):
        dt_h = (t_seconds[i] - t_seconds[i - 1]) / 3600.0
        # use absolute power (energy used or recovered counted separately)
        incremental = abs(p[i - 1]) * dt_h
        cumulative.append(cumulative[-1] + incremental)
    return times, cumulative


# ==================== MAIN APP ====================


def main():
    """Main application"""
    st.markdown("## ⚡ TATA NEXON EV DIGITAL TWIN PROTOTYPE")
    st.caption("Real-time monitoring & analytics (40.5 kWh, 312 km range)")

    initialize_session_state()

    ev = st.session_state.ev_twin

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(ev.config["model"])
    with col2:
        st.metric("Vehicle ID", ev.vehicle_id)

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Vehicle Controls")

        col1b, col2b = st.columns(2)
        with col1b:
            if st.button(
                "▶️ Start" if not st.session_state.simulation_running else "⏸️ Pause",
                use_container_width=True,
                key="start_pause",
            ):
                st.session_state.simulation_running = not st.session_state.simulation_running
                st.rerun()

        with col2b:
            if st.button("🔄 Reset", use_container_width=True, key="reset"):
                st.session_state.ev_twin = TataNexonEVDigitalTwin(dataset=None)
                st.session_state.simulation_running = False
                initialize_session_state()
                st.rerun()

        st.markdown("---")

        st.subheader("🚗 Driving Mode")
        mode_selection = st.selectbox(
            "Select Mode", [m.value for m in DrivingMode], key="driving_mode"
        )

        if mode_selection == DrivingMode.ECO.value:
            ev.driving_mode = DrivingMode.ECO
            ev.powertrain.set_driving_mode(DrivingMode.ECO)
            st.info("🌱 Eco Mode: Max efficiency")
        elif mode_selection == DrivingMode.SPORT.value:
            ev.driving_mode = DrivingMode.SPORT
            ev.powertrain.set_driving_mode(DrivingMode.SPORT)
            st.success("🏁 Sport Mode: Full power")
        else:
            ev.driving_mode = DrivingMode.CITY
            ev.powertrain.set_driving_mode(DrivingMode.CITY)
            st.info("🏙️ City Mode: Balanced")

        st.markdown("---")

        st.subheader("🎮 Driver Inputs (Note: disabled while charging)")
        # If charging active -> lock acceleration to 0 and allow only mechanical braking
        if ev.charging.charging_active:
            st.info(
                "⚡ Charging active: propulsion disabled. Accelerator locked to 0. Regen disabled."
            )
            _ = st.slider(
                "Accelerator (%)",
                0,
                100,
                0,
                5,
                key="accel_display",
                disabled=True,
            )
            ev.accelerator_position = 0.0
            ev.brake_position = st.slider(
                "Brake Pedal (%) (mechanical only)",
                0,
                100,
                0,
                5,
                key="brake_while_charging",
            )
        else:
            accel_val = st.slider(
                "Accelerator (%)",
                0,
                100,
                int(ev.accelerator_position),
                5,
                key="accel",
            )
            ev.accelerator_position = float(accel_val)

            # Compute allowed brake range (avoid slider exception by branching)
            max_brake_allowed = max(0, int(round(100 - ev.accelerator_position)))
            if max_brake_allowed <= 0:
                # Can't use st.slider(0,0,...) -> use small 0-1 disabled slider and show info
                st.info("Accelerator at 100% — braking disabled.")
                _ = st.slider(
                    "Brake Pedal (%)",
                    0,
                    1,
                    0,
                    1,
                    key="brake_disabled",
                    disabled=True,
                )
                ev.brake_position = 0.0
            else:
                # ensure default within bounds
                default_brake = (
                    int(ev.brake_position)
                    if 0 <= ev.brake_position <= max_brake_allowed
                    else 0
                )
                ev.brake_position = float(
                    st.slider(
                        "Brake Pedal (%)",
                        0,
                        max_brake_allowed,
                        default_brake,
                        5,
                        key="brake",
                    )
                )

        st.markdown("---")

        st.subheader("🔋 Regenerative Braking")
        regen_labels = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}
        ev.braking.regen_level = st.select_slider(
            "Regen Level",
            options=[0, 1, 2, 3],
            value=ev.braking.regen_level,
            format_func=lambda x: regen_labels[x],
            key="regen",
        )

        st.markdown("---")

        st.subheader("🌡️ Environment")
        ev.ambient_temperature = st.slider(
            "Ambient Temp (°C)",
            0,
            50,
            int(ev.ambient_temperature),
            1,
            key="temp",
        )

        st.markdown("---")

        st.subheader("⚡ Charging")
        protocol = st.selectbox(
            "Charging Type", [p.value for p in ChargingProtocol], key="protocol"
        )

        if st.button("🔌 Start Charging", key="start_charge"):
            # Block charging if vehicle is moving or accelerator is pressed
            current_speed = ev.dynamics.speed_kmh
            current_accel = ev.accelerator_position
            if ev.shutdown:
                st.error(
                    "Cannot start charging: vehicle in emergency shutdown (thermal runaway)."
                )
            elif current_speed > 0.1:
                st.error(
                    f"Cannot start charging while vehicle speed is {current_speed:.1f} km/h. Stop the vehicle first."
                )
            elif current_accel > 0.1:
                st.error(
                    f"Cannot start charging while accelerator is at {current_accel:.1f}%. Release the accelerator first."
                )
            else:
                result = ev.charging.initiate_charging(
                    protocol, 100, ev.bms.soc, ev.config["capacity_kwh"]
                )
                st.success(f"Charging started: {result['power_kw']} kW")
                # lock acceleration will be enforced by update loop

        if st.button("⏹️ Stop Charging", key="stop_charge"):
            ev.charging.stop_charging()
            st.info("Charging stopped")

    # Real-time KPIs (auto-updates every 0.2 s)
    render_realtime_kpis()

    st.markdown("---")

    state = st.session_state.current_state
    ev = st.session_state.ev_twin

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔋 Battery", "⚡ Powertrain", "🛑 Braking", "🔌 Charging", "📈 Analytics"]
    )

    # ========== BATTERY TAB ==========
    with tab1:
        st.subheader("Battery Management System")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Battery Status")
            df = pd.DataFrame(
                {
                    "Parameter": [
                        "Voltage",
                        "Current",
                        "Temperature",
                        "Power (Battery)",
                        "Coolant Temp",
                    ],
                    "Value": [
                        f"{state['bms']['voltage']:.1f} V",
                        f"{state['bms']['current']:.1f} A",
                        f"{state['bms']['temperature']:.1f} °C",
                        f"{state['bms']['power']:.1f} kW",
                        f"{state['bms']['coolant_temp']:.1f} °C",
                    ],
                }
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

        with c2:
            st.markdown("#### Thermal System")
            thermal_df = pd.DataFrame(
                {
                    "System": [
                        "Liquid Cooling",
                        "Battery Heating",
                        "Temp Limit",
                        "Status",
                    ],
                    "State": [
                        "🟢 Active"
                        if state["bms"]["cooling_active"]
                        else "⚪ Standby",
                        "🟢 Active"
                        if state["bms"]["heating_active"]
                        else "⚪ Standby",
                        "Max 60°C",
                        "🟢 Normal"
                        if 15 <= state["bms"]["temperature"] <= 50
                        else "⚠️ Sub-optimal",
                    ],
                }
            )
            st.dataframe(thermal_df, hide_index=True, use_container_width=True)

        with c3:
            st.markdown("#### Specifications")
            st.info(
                f"**Model:** {ev.config['model']}\n\n"
                f"**Capacity:** {ev.config['capacity_kwh']} kWh  \n"
                f"**Usable:** {ev.config['usable_capacity_kwh']} kWh  \n"
                f"**Nominal Voltage:** {ev.config['nominal_voltage']} V  \n"
                f"**Max Temp:** 60°C (warning), Thermal Runaway: 120°C"
            )

        # SOC-OCV Curve
        soc_curve = np.linspace(0, 100, 100)
        ocv_curve = [ev.bms._soc_to_ocv(s) for s in soc_curve]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=soc_curve, y=ocv_curve, mode="lines", name="OCV Curve")
        )
        fig.add_trace(
            go.Scatter(
                x=[state["bms"]["soc"]],
                y=[state["bms"]["voltage"]],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="Current",
            )
        )
        fig.update_layout(
            xaxis_title="SOC (%)",
            yaxis_title="Voltage (V)",
            height=300,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ========== POWERTRAIN TAB ==========
    with tab2:
        st.subheader("Powertrain System")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"#### Motor ({state['powertrain']['driving_mode']})")
            motor_df = pd.DataFrame(
                {
                    "Parameter": ["RPM", "Torque", "Power (Shaft)", "Max Power"],
                    "Value": [
                        f"{state['powertrain']['motor_rpm']:.0f}",
                        f"{state['powertrain']['motor_torque']:.1f} Nm",
                        f"{state['powertrain']['power_kw']:.1f} kW",
                        f"{ev.config['max_power_kw']} kW",
                    ],
                }
            )
            st.dataframe(motor_df, hide_index=True, use_container_width=True)

        with c2:
            st.markdown("#### Vehicle Dynamics")
            dyn_df = pd.DataFrame(
                {
                    "Parameter": ["Speed", "Acceleration", "Distance"],
                    "Value": [
                        f"{state['dynamics']['speed_kmh']:.1f} km/h",
                        f"{state['dynamics']['acceleration_mps2']:.2f} m/s²",
                        f"{state['dynamics']['distance_km']:.2f} km",
                    ],
                }
            )
            st.dataframe(dyn_df, hide_index=True, use_container_width=True)

    # ========== BRAKING TAB ==========
    with tab3:
        st.subheader("Braking System")
        c1, c2 = st.columns(2)

        with c1:
            regen_display = (
                "⚡ Charging"
                if ev.charging.charging_active
                else ("🟢 Active" if state["braking"]["regen_active"] else "⚪ Off")
            )
            brake_df = pd.DataFrame(
                {
                    "Parameter": ["Pressure", "Regen Braking", "Mechanical", "ABS", "Regen Power"],
                    "Value": [
                        f"{state['braking']['brake_pressure']:.0f}%",
                        regen_display,
                        "🟢 Active"
                        if state["braking"]["mechanical_active"]
                        else "⚪ Off",
                        "🟢 Active"
                        if state["braking"]["abs_active"]
                        else "⚪ Standby",
                        f"{abs(state['braking']['regen_power_kw']):.1f} kW"
                        if not ev.charging.charging_active
                        else "—",
                    ],
                }
            )
            st.dataframe(brake_df, hide_index=True, use_container_width=True)

        with c2:
            recovery_df = pd.DataFrame(
                {
                    "Metric": ["Recovered", "Front Brake Temp", "Rear Brake Temp"],
                    "Value": [
                        f"{state['braking']['energy_recovered']:.2f} kWh",
                        f"{state['braking']['brake_temp_front']:.1f}°C",
                        f"{state['braking']['brake_temp_rear']:.1f}°C",
                    ],
                }
            )
            st.dataframe(recovery_df, hide_index=True, use_container_width=True)

    # ========== CHARGING TAB ==========
    with tab4:
        st.subheader("Charging System")
        if ev.charging.charging_active:
            st.success("🔌 Charging Active")
            charging_state = ev.charging.update_charging(ev.bms.soc, 100, ev.bms.temperature)
            if charging_state.get("charging_active", False):
                charge_df = pd.DataFrame(
                    {
                        "Parameter": ["Protocol", "Power", "SOC", "Time Remaining"],
                        "Value": [
                            ev.charging.charging_protocol,
                            f"{charging_state['actual_power_kw']:.1f} kW",
                            f"{ev.bms.soc:.1f}%",
                            f"{charging_state['time_remaining_min']:.1f} min",
                        ],
                    }
                )
                st.dataframe(charge_df, hide_index=True, use_container_width=True)
        else:
            st.info("⚪ Not Charging")

        charge_options = pd.DataFrame(
            {
                "Type": ["DC Fast", "AC Fast", "Home"],
                "Power": ["50 kW", "7.2 kW", "3.3 kW"],
                "0–80% (approx)": ["~50 min", "~6 hrs", "~12 hrs"],
            }
        )
        st.markdown("#### Typical Charging Options")
        st.dataframe(charge_options, hide_index=True, use_container_width=True)

    # ========== ANALYTICS TAB ==========
    with tab5:
        st.subheader("Vehicle Analytics (Digital Twin)")

        if len(ev.history["soc"]) > 5:
            # Basic time-series selector
            plot_vars = st.multiselect(
                "Select metrics for time-series plot",
                ["SOC (%)", "Speed (km/h)", "Power (kW, battery)", "Temperature (°C)"],
                default=["SOC (%)", "Power (kW, battery)"],
                key="metrics",
            )

            if plot_vars:
                fig = go.Figure()
                data_map = {
                    "SOC (%)": ev.history["soc"][-200:],
                    "Speed (km/h)": ev.history["speed"][-200:],
                    "Power (kW, battery)": ev.history["power"][-200:],
                    "Temperature (°C)": ev.history["temperature"][-200:],
                }
                time_slice = ev.history["time"][-len(next(iter(data_map.values()))):]

                for var in plot_vars:
                    fig.add_trace(
                        go.Scatter(
                            x=time_slice,
                            y=data_map[var],
                            mode="lines",
                            name=var,
                        )
                    )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=350,
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Twin Analytics (Extended OEM-ready)")

            # Row 1: SOC vs Time and Cumulative Energy
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                fig_soc = go.Figure()
                fig_soc.add_trace(
                    go.Scatter(
                        x=ev.history["time"],
                        y=ev.history["soc"],
                        mode="lines",
                        name="SOC (%)",
                    )
                )
                fig_soc.update_layout(
                    title="SOC vs Time",
                    xaxis_title="Time",
                    yaxis_title="SOC (%)",
                    height=300,
                    template="plotly_white",
                )
                st.plotly_chart(fig_soc, use_container_width=True)

            with r1c2:
                times, cumulative_energy = compute_cumulative_energy_from_history(ev)
                fig_energy = go.Figure()
                fig_energy.add_trace(
                    go.Scatter(
                        x=times,
                        y=cumulative_energy,
                        mode="lines+markers",
                        name="Cumulative Energy (kWh)",
                    )
                )
                fig_energy.update_layout(
                    title="Cumulative Energy (kWh) from Battery Power History",
                    xaxis_title="Time",
                    yaxis_title="Energy (kWh)",
                    height=300,
                    template="plotly_white",
                )
                st.plotly_chart(fig_energy, use_container_width=True)

            # Row 2: Recovered energy & SOC-Speed heatmap
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                fig_recovered = go.Figure()
                fig_recovered.add_trace(
                    go.Scatter(
                        x=ev.history["time"],
                        y=ev.history["energy_recovered"],
                        mode="lines+markers",
                        name="Recovered Energy (kWh)",
                    )
                )
                fig_recovered.update_layout(
                    title="Regenerative Energy Recovered (kWh)",
                    xaxis_title="Time",
                    yaxis_title="kWh",
                    height=300,
                    template="plotly_white",
                )
                st.plotly_chart(fig_recovered, use_container_width=True)

            with r2c2:
                # SOC vs Speed 2D histogram / density
                soc_arr = np.array(ev.history["soc"])
                spd_arr = np.array(ev.history["speed"])
                # create heatmap bins
                try:
                    heatmap, xedges, yedges = np.histogram2d(
                        soc_arr,
                        spd_arr,
                        bins=[20, 20],
                        range=[[0, 100], [0, max(1.0, spd_arr.max() + 1)]],
                    )
                    heatmap = heatmap.T  # transpose for correct orientation
                    fig_heat = go.Figure(
                        data=go.Heatmap(
                            z=heatmap,
                            x=(xedges[:-1] + xedges[1:]) / 2.0,
                            y=(yedges[:-1] + yedges[1:]) / 2.0,
                            colorbar=dict(title="Count"),
                        )
                    )
                    fig_heat.update_layout(
                        title="SOC vs Speed Density",
                        xaxis_title="SOC (%)",
                        yaxis_title="Speed (km/h)",
                        height=300,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                except Exception:
                    st.info("Not enough variation for SOC vs Speed heatmap.")

            # Row 3: Temp rolling mean & distribution, Power vs Speed scatter
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                temps = pd.Series(ev.history["temperature"])
                times_pd = pd.to_datetime(ev.history["time"])
                if len(temps) >= 3:
                    rolling = temps.rolling(window=min(10, len(temps))).mean()
                    fig_temp = go.Figure()
                    fig_temp.add_trace(
                        go.Scatter(
                            x=times_pd,
                            y=temps,
                            mode="markers+lines",
                            name="Temperature",
                        )
                    )
                    fig_temp.add_trace(
                        go.Scatter(
                            x=times_pd,
                            y=rolling,
                            mode="lines",
                            name="Rolling Mean",
                        )
                    )
                    fig_temp.update_layout(
                        title="Battery Temperature (with rolling mean)",
                        xaxis_title="Time",
                        yaxis_title="Temperature (°C)",
                        height=300,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                else:
                    st.info("Collect more samples for rolling temperature analysis.")
                # Add a box plot to show distribution
                fig_box = go.Figure()
                fig_box.add_trace(
                    go.Box(y=temps, name="Temperature distribution")
                )
                fig_box.update_layout(
                    title="Temperature Distribution",
                    height=250,
                    template="plotly_white",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with r3c2:
                # Power vs Speed scatter (battery-side power)
                fig_ps = go.Figure()
                fig_ps.add_trace(
                    go.Scatter(
                        x=ev.history["speed"],
                        y=ev.history["power"],
                        mode="markers",
                        opacity=0.7,
                        name="Battery Power vs Speed",
                    )
                )
                fig_ps.update_layout(
                    title="Battery Power (kW) vs Speed (km/h)",
                    xaxis_title="Speed (km/h)",
                    yaxis_title="Battery Power (kW)",
                    height=300,
                    template="plotly_white",
                )
                st.plotly_chart(fig_ps, use_container_width=True)

                # SOC vs Temperature scatter (for thermal-health correlation)
                fig_st = go.Figure()
                fig_st.add_trace(
                    go.Scatter(
                        x=ev.history["soc"],
                        y=ev.history["temperature"],
                        mode="markers",
                        opacity=0.7,
                        name="SOC vs Temp",
                    )
                )
                fig_st.update_layout(
                    title="SOC vs Battery Temperature",
                    xaxis_title="SOC (%)",
                    yaxis_title="Temperature (°C)",
                    height=300,
                    template="plotly_white",
                )
                st.plotly_chart(fig_st, use_container_width=True)

            # Summary stats
            st.markdown("### Summary Statistics")
            col1a, col2a, col3a, col4a = st.columns(4)
            with col1a:
                st.metric(
                    "Avg |Battery Power|",
                    f"{np.mean([abs(p) for p in ev.history['power']]):.1f} kW",
                )
            with col2a:
                st.metric("Max Speed", f"{np.max(ev.history['speed']):.1f} km/h")
            with col3a:
                st.metric("Energy Used", f"{ev.bms.energy_consumed:.2f} kWh")
            with col4a:
                st.metric("Recovered", f"{ev.braking.energy_recovered_kwh:.2f} kWh")

            # CSV export
            if st.button("📥 Prepare CSV for Download", key="prepare_csv"):
                df_export = pd.DataFrame(
                    {
                        "Time": ev.history["time"],
                        "SOC": ev.history["soc"],
                        "Speed": ev.history["speed"],
                        "BatteryPower_kW": ev.history["power"],
                        "Temp": ev.history["temperature"],
                        "Recovered_kWh": ev.history["energy_recovered"],
                    }
                )
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "Download Digital Twin Log",
                    csv,
                    f"nexon_digital_twin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="dl_btn",
                )
        else:
            st.info("📊 Start simulation to collect data")

    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #0D47A1;'>
        <p><strong>TATA NEXON EV DIGITAL TWIN PROTOTYPE</strong></p>
        <p style='font-size: 0.9em;'>
        ⚡ Real-time auto-refresh every 0.2 seconds | Max Battery Temp: 60°C (warning) | Thermal Runaway >= 120°C
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
