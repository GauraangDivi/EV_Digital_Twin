# tata_nexon_ev_realtime.py
"""
Tata Nexon EV Digital Twin - Real-time RT-Operation Version
Smooth continuous updates using @st.fragment
Max battery temperature: 60°C
"""


import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from enum import Enum


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Tata Nexon EV RT Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)


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


    def _soc_to_ocv(self, soc):
        """OCV lookup table"""
        soc_points = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ocv_points = np.array([260, 285, 300, 310, 320, 325, 330, 340, 350, 365, 380])
        return np.interp(soc, soc_points, ocv_points)


    def estimate_soc_coulomb_counting(self, current_a, dt_hours):
        capacity_ah = self.config['capacity_kwh'] * 1000 / self.config['nominal_voltage']
        delta_soc = (current_a * dt_hours / capacity_ah) * 100
        self.soc = np.clip(self.soc - delta_soc, 0, 100)
        return self.soc


    def estimate_soc_kalman_filter(self, measured_voltage):
        prediction = self.kf_estimate
        prediction_error = self.kf_error_covariance + self.process_noise


        ocv_expected = self._soc_to_ocv(prediction)
        innovation = measured_voltage - ocv_expected


        kalman_gain = prediction_error / (prediction_error + self.measurement_noise)
        self.kf_estimate = prediction + kalman_gain * innovation
        self.kf_error_covariance = (1 - kalman_gain) * prediction_error


        self.kf_estimate = np.clip(self.kf_estimate, 0, 100)
        return self.kf_estimate


    def estimate_soh(self):
        capacity_fade = 100 * np.exp(-0.00003 * self.cycle_count)
        self.soh = np.clip(capacity_fade, 70, 100)
        return self.soh


    def advanced_thermal_management(self, ambient_temp, power_dissipation):
        """
        Manages the battery temperature, keeping it within the optimal range (15-35°C)
        and enforcing a strict maximum limit of 60°C.
        """
        heat_generation = (self.current ** 2) * 0.015  # Simplified heat generation model

        # The nominal operating range is between 15°C and 35°C.
        # Below 15°C, the battery heater is activated.
        # Above 35°C, the liquid cooling system is activated.
        if self.temperature > 35:
            self.cooling_active = True
            self.heating_active = False
            cooling_power = 1.5  # Cooling system efficacy factor
            heat_dissipation = cooling_power * (self.temperature - ambient_temp)
        elif self.temperature < 15:
            self.heating_active = True
            self.cooling_active = False
            heating_power = 0.8  # Heating system efficacy factor
            # Heat is added, so dissipation is negative
            heat_dissipation = -heating_power * (ambient_temp - self.temperature)
        else:
            # Within the nominal range, only passive heat dissipation occurs.
            self.cooling_active = False
            self.heating_active = False
            heat_dissipation = 0.3 * (self.temperature - ambient_temp)

        thermal_mass = 150  # Represents the battery pack's resistance to temperature change
        dt = 1  # Simulation time step
        
        # Calculate the change in temperature for this time step
        temp_change = (heat_generation - heat_dissipation) * dt / thermal_mass
        
        # Calculate the potential new temperature
        potential_temp = self.temperature + temp_change
        
        # Enforce the absolute maximum temperature of 60°C.
        # The battery temperature cannot exceed this limit. If the calculation
        # suggests a higher temperature, it's capped at 60°C. The cooling
        # system (already active if temp > 35) will then work to reduce it.
        self.temperature = min(potential_temp, 60.0)

        # Update the coolant temperature based on the state of the cooling system.
        self.coolant_temp = self.temperature - 5 if self.cooling_active else self.temperature

        return self.temperature, self.cooling_active, self.heating_active


    def safety_check(self):
        warnings = []
        errors = []


        if self.voltage > self.config['max_voltage'] * 1.05:
            errors.append("CRITICAL: Overvoltage detected")
        elif self.voltage > self.config['max_voltage']:
            warnings.append("Warning: High voltage")


        # The hard limit is 60°C. A warning is issued above 50°C.
        if self.temperature >= 60:
            errors.append("CRITICAL: Battery at max limit (60°C)")
        elif self.temperature > 50:
            warnings.append("Warning: High temperature (>50°C)")


        if self.temperature < 0:
            errors.append("CRITICAL: Battery too cold")
        elif self.temperature < 10:
            warnings.append("Warning: Low temperature")


        if self.soc < 3:
            errors.append("CRITICAL: Battery critically low")
        elif self.soc < 10:
            warnings.append("Warning: Low battery")


        return warnings, errors


    def update(self, current_a, ambient_temp, dt_seconds=1.0):
        dt_hours = dt_seconds / 3600.0


        self.current = current_a
        self.voltage = self._soc_to_ocv(self.soc)
        self.power = (self.voltage * self.current) / 1000.0


        soc_coulomb = self.estimate_soc_coulomb_counting(current_a, dt_hours)
        soc_kalman = self.estimate_soc_kalman_filter(self.voltage)
        self.soc = 0.7 * soc_kalman + 0.3 * soc_coulomb


        if abs(current_a) > 10:
            self.cycle_count += dt_hours * 0.00005
        self.estimate_soh()


        self.temperature, cooling_active, heating_active = self.advanced_thermal_management(
            ambient_temp, abs(self.power)
        )


        warnings, errors = self.safety_check()


        self.energy_consumed += abs(self.power * dt_hours)


        return {
            'soc': self.soc,
            'soh': self.soh,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'power': self.power,
            'cooling_active': cooling_active,
            'heating_active': heating_active,
            'coolant_temp': self.coolant_temp,
            'warnings': warnings,
            'errors': errors
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


        if vehicle_speed_kmh > 20:
            regen_effectiveness = 1.0
        elif vehicle_speed_kmh > 10:
            regen_effectiveness = 0.7
        elif vehicle_speed_kmh > 5:
            regen_effectiveness = 0.3
        else:
            regen_effectiveness = 0.0


        regen_multipliers = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9}
        regen_factor = regen_multipliers[self.regen_level] * regen_effectiveness


        regen_force_pct = brake_pedal_pct * regen_factor
        mechanical_force_pct = brake_pedal_pct * (1 - regen_factor)


        self.regen_brake_active = regen_force_pct > 5
        self.mechanical_brake_active = mechanical_force_pct > 5
        self.abs_active = brake_pedal_pct > 80 and vehicle_speed_kmh > 30


        return regen_force_pct, mechanical_force_pct


    def calculate_regen_power(self, vehicle_speed_kmh, regen_force_pct):
        if not self.regen_brake_active:
            return 0.0


        max_regen_power = 35
        speed_factor = min(vehicle_speed_kmh / 100, 1.0)
        regen_power_kw = -max_regen_power * (regen_force_pct / 100) * speed_factor * 0.85


        return regen_power_kw


    def update_brake_temperature(self, mechanical_force_pct, vehicle_speed_kmh, ambient_temp):
        if self.mechanical_brake_active:
            heat_gen_front = mechanical_force_pct * vehicle_speed_kmh * 0.02
            heat_gen_rear = mechanical_force_pct * vehicle_speed_kmh * 0.015
        else:
            heat_gen_front = 0
            heat_gen_rear = 0


        cooling_rate = 0.5 + (vehicle_speed_kmh * 0.01)


        self.brake_temp_front += (heat_gen_front - cooling_rate * (self.brake_temp_front - ambient_temp)) * 0.1
        self.brake_temp_rear += (heat_gen_rear - cooling_rate * (self.brake_temp_rear - ambient_temp)) * 0.1


        self.brake_temp_front = max(ambient_temp, self.brake_temp_front)
        self.brake_temp_rear = max(ambient_temp, self.brake_temp_rear)


    def update(self, brake_pedal_pct, vehicle_speed_kmh, ambient_temp, dt_seconds=1.0):
        regen_force, mech_force = self.calculate_braking_force(brake_pedal_pct, vehicle_speed_kmh)
        regen_power = self.calculate_regen_power(vehicle_speed_kmh, regen_force)


        if regen_power < 0:
            self.energy_recovered_kwh += abs(regen_power) * (dt_seconds / 3600.0)


        self.update_brake_temperature(mech_force, vehicle_speed_kmh, ambient_temp)


        return {
            'brake_pressure': self.brake_pressure,
            'regen_active': self.regen_brake_active,
            'mechanical_active': self.mechanical_brake_active,
            'abs_active': self.abs_active,
            'regen_power_kw': regen_power,
            'regen_level': self.regen_level,
            'energy_recovered': self.energy_recovered_kwh,
            'brake_temp_front': self.brake_temp_front,
            'brake_temp_rear': self.brake_temp_rear
        }


# ==================== POWERTRAIN ====================


class TataNexonPowertrain:
    """Tata Nexon EV Powertrain"""


    def __init__(self, config, driving_mode=DrivingMode.CITY):
        self.config = config
        self.driving_mode = driving_mode
        self.motor_rpm = 0
        self.motor_torque = 0
        self.motor_efficiency = 0.92
        self.inverter_efficiency = 0.96


    def set_driving_mode(self, mode):
        self.driving_mode = mode


    def calculate_motor_speed(self, vehicle_speed_kmh):
        wheel_diameter_m = 0.665
        gear_ratio = 10.5
        wheel_rpm = (vehicle_speed_kmh * 1000 / 60) / (np.pi * wheel_diameter_m)
        motor_rpm = wheel_rpm * gear_ratio
        return motor_rpm


    def calculate_torque_demand(self, accelerator_pct, vehicle_speed_kmh):
        max_torque = self.config['max_torque_nm']


        if self.driving_mode == DrivingMode.ECO:
            base_torque = (accelerator_pct / 100) * max_torque * 0.80
        elif self.driving_mode == DrivingMode.CITY:
            base_torque = (accelerator_pct / 100) * max_torque * 0.90
        elif self.driving_mode == DrivingMode.SPORT:
            base_torque = (accelerator_pct / 100) * max_torque * 1.0
        else:
            base_torque = (accelerator_pct / 100) * max_torque


        if vehicle_speed_kmh > 100:
            derating_factor = 1.0 - (vehicle_speed_kmh - 100) / 120
            base_torque *= max(0.4, derating_factor)


        self.motor_torque = base_torque
        return self.motor_torque


    def calculate_power(self, motor_rpm):
        power_w = (self.motor_torque * motor_rpm * 2 * np.pi) / 60
        power_kw = power_w / 1000


        if self.motor_torque > 0:
            power_kw = power_kw / (self.motor_efficiency * self.inverter_efficiency)


        return power_kw


    def update(self, accelerator_pct, vehicle_speed_kmh):
        self.motor_rpm = self.calculate_motor_speed(vehicle_speed_kmh)
        self.motor_torque = self.calculate_torque_demand(accelerator_pct, vehicle_speed_kmh)
        power_kw = self.calculate_power(self.motor_rpm)


        return {
            'motor_rpm': self.motor_rpm,
            'motor_torque': self.motor_torque,
            'power_kw': power_kw,
            'driving_mode': self.driving_mode.value
        }


# ==================== VEHICLE DYNAMICS ====================


class TataNexonDynamics:
    """Vehicle dynamics"""


    def __init__(self, config):
        self.config = config
        self.speed_kmh = 0
        self.acceleration_mps2 = 0
        self.distance_km = 0


    def update(self, motor_torque_nm, brake_force_pct, dt_seconds=1.0):
        mass_kg = 1400
        wheel_radius_m = 0.3325
        gear_ratio = 10.5
        air_drag_coeff = 0.33
        frontal_area_m2 = 2.4
        rolling_resistance_coeff = 0.012


        wheel_force_n = (motor_torque_nm * gear_ratio) / wheel_radius_m
        brake_force_n = brake_force_pct * 50
        drag_force_n = 0.5 * 1.225 * air_drag_coeff * frontal_area_m2 * (self.speed_kmh / 3.6) ** 2
        rolling_resistance_n = rolling_resistance_coeff * mass_kg * 9.81


        net_force_n = wheel_force_n - brake_force_n - drag_force_n - rolling_resistance_n
        self.acceleration_mps2 = net_force_n / mass_kg


        delta_speed_mps = self.acceleration_mps2 * dt_seconds
        self.speed_kmh += delta_speed_mps * 3.6
        self.speed_kmh = max(0, min(self.speed_kmh, 140))


        self.distance_km += (self.speed_kmh / 3600) * dt_seconds


        return {
            'speed_kmh': self.speed_kmh,
            'acceleration_mps2': self.acceleration_mps2,
            'distance_km': self.distance_km
        }


# ==================== CHARGING SYSTEM ====================


class TataNexonChargingSystem:
    """Charging System"""


    def __init__(self):
        self.charging_active = False
        self.charging_protocol = None
        self.charging_power_kw = 0
        self.charging_current_a = 0
        self.charge_time_remaining_min = 0


    def initiate_charging(self, protocol, target_soc, current_soc, battery_capacity_kwh):
        self.charging_protocol = protocol
        self.charging_active = True


        if "50kW" in protocol:
            self.charging_power_kw = 50
            self.charging_current_a = 125
        elif "7.2kW" in protocol:
            self.charging_power_kw = 7.2
            self.charging_current_a = 32
        else:
            self.charging_power_kw = 3.3
            self.charging_current_a = 16


        energy_needed = (target_soc - current_soc) / 100 * battery_capacity_kwh
        self.charge_time_remaining_min = (energy_needed / self.charging_power_kw) * 60


        return {
            'status': 'Charging initiated',
            'protocol': self.charging_protocol,
            'power_kw': self.charging_power_kw,
            'time_remaining_min': self.charge_time_remaining_min
        }


    def update_charging(self, current_soc, target_soc, battery_temp, dt_seconds=1.0):
        if not self.charging_active:
            return {'charging_active': False}


        if battery_temp > 40:
            temp_factor = 0.7
        elif battery_temp < 15:
            temp_factor = 0.8
        else:
            temp_factor = 1.0


        if current_soc > 80:
            soc_factor = 1.0 - ((current_soc - 80) / 20) * 0.6
        else:
            soc_factor = 1.0


        actual_power = self.charging_power_kw * temp_factor * soc_factor


        if current_soc >= target_soc:
            self.stop_charging()
            return {'charging_active': False, 'status': 'Charging complete'}


        return {
            'charging_active': True,
            'actual_power_kw': actual_power,
            'time_remaining_min': self.charge_time_remaining_min
        }


    def stop_charging(self):
        self.charging_active = False
        self.charging_power_kw = 0

# ==================== DIGITAL TWIN ====================


class TataNexonEVDigitalTwin:
    """Complete Tata Nexon EV Digital Twin"""


    def __init__(self, vehicle_id="TN-NEXON-001"):
        self.vehicle_id = vehicle_id


        self.config = {
            'model': 'Tata Nexon EV (40.5 kWh)',
            'capacity_kwh': 40.5,
            'usable_capacity_kwh': 38.8,
            'nominal_voltage': 325,
            'max_voltage': 380,
            'min_voltage': 260,
            'max_discharge_current': 300,
            'max_power_kw': 105,
            'max_torque_nm': 250,
            'range_km': 312,
            'weight_kg': 1400,
            'top_speed_kmh': 140,
            'acceleration_0_100': 9.9
        }


        self.bms = TataNexonBMS(self.config)
        self.powertrain = TataNexonPowertrain(self.config)
        self.braking = TataNexonBrakingSystem()
        self.dynamics = TataNexonDynamics(self.config)
        self.charging = TataNexonChargingSystem()


        self.ambient_temperature = 30.0
        self.accelerator_position = 0
        self.brake_position = 0
        self.driving_mode = DrivingMode.CITY


        self.history = {
            'time': [],
            'soc': [],
            'soh': [],
            'speed': [],
            'power': [],
            'temperature': [],
            'energy_recovered': []
        }


    def update(self, dt_seconds=0.5):
        """Main update loop"""
        powertrain_state = self.powertrain.update(
            self.accelerator_position,
            self.dynamics.speed_kmh
        )


        braking_state = self.braking.update(
            self.brake_position,
            self.dynamics.speed_kmh,
            self.ambient_temperature,
            dt_seconds
        )


        motor_power = powertrain_state['power_kw']
        regen_power = braking_state['regen_power_kw']


        if motor_power != 0 or regen_power != 0:
            total_power = motor_power + regen_power
            battery_current = (total_power * 1000) / max(self.bms.voltage, 1)
        else:
            battery_current = 0


        if self.charging.charging_active:
            charging_state = self.charging.update_charging(
                self.bms.soc, 100, self.bms.temperature, dt_seconds
            )
            if charging_state['charging_active']:
                battery_current = -charging_state['actual_power_kw'] * 1000 / max(self.bms.voltage, 1)


        bms_state = self.bms.update(battery_current, self.ambient_temperature, dt_seconds)


        brake_force = braking_state['brake_pressure'] * (1 if braking_state['mechanical_active'] else 0.3)
        dynamics_state = self.dynamics.update(
            powertrain_state['motor_torque'],
            brake_force,
            dt_seconds
        )


        self.history['time'].append(datetime.now())
        self.history['soc'].append(bms_state['soc'])
        self.history['soh'].append(bms_state['soh'])
        self.history['speed'].append(dynamics_state['speed_kmh'])
        self.history['power'].append(bms_state['power'])
        self.history['temperature'].append(bms_state['temperature'])
        self.history['energy_recovered'].append(braking_state['energy_recovered'])


        max_history = 500
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]


        remaining_energy = (bms_state['soc'] / 100) * self.config['usable_capacity_kwh']
        avg_consumption = 0.124
        estimated_range = remaining_energy / avg_consumption


        return {
            'bms': bms_state,
            'powertrain': powertrain_state,
            'braking': braking_state,
            'dynamics': dynamics_state,
            'estimated_range_km': estimated_range
        }


# ==================== STREAMLIT UI ====================


def initialize_session_state():
    """Initialize session state"""
    if 'ev_twin' not in st.session_state:
        st.session_state.ev_twin = TataNexonEVDigitalTwin()
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    if 'current_state' not in st.session_state:
        ev = st.session_state.ev_twin
        remaining_energy = (ev.bms.soc / 100) * ev.config['usable_capacity_kwh']
        st.session_state.current_state = {
            'bms': {
                'soc': ev.bms.soc,
                'soh': ev.bms.soh,
                'voltage': ev.bms.voltage,
                'current': ev.bms.current,
                'temperature': ev.bms.temperature,
                'power': ev.bms.power,
                'cooling_active': False,
                'heating_active': False,
                'coolant_temp': ev.bms.coolant_temp,
                'warnings': [],
                'errors': []
            },
            'powertrain': {
                'motor_rpm': 0,
                'motor_torque': 0,
                'power_kw': 0,
                'driving_mode': ev.driving_mode.value
            },
            'braking': {
                'brake_pressure': 0,
                'regen_active': False,
                'mechanical_active': False,
                'abs_active': False,
                'regen_power_kw': 0,
                'regen_level': ev.braking.regen_level,
                'energy_recovered': ev.braking.energy_recovered_kwh,
                'brake_temp_front': ev.braking.brake_temp_front,
                'brake_temp_rear': ev.braking.brake_temp_rear
            },
            'dynamics': {
                'speed_kmh': ev.dynamics.speed_kmh,
                'acceleration_mps2': 0,
                'distance_km': ev.dynamics.distance_km
            },
            'estimated_range_km': remaining_energy / 0.124
        }


@st.fragment(run_every="0.5s")
def render_realtime_kpis():
    """Real-time KPI updates - only this section refreshes"""
    ev = st.session_state.ev_twin


    # Update simulation if running
    if st.session_state.simulation_running:
        current_time = time.time()
        dt = current_time - st.session_state.last_update
        state = ev.update(dt_seconds=min(dt, 1.0))
        st.session_state.last_update = current_time
        st.session_state.current_state = state
    else:
        state = st.session_state.current_state


    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        st.metric("🔋 Battery SOC", f"{state['bms']['soc']:.1f}%",
                    delta=f"SOH: {state['bms']['soh']:.1f}%")


    with col2:
        power_label = "Regen" if state['bms']['power'] < 0 else "Consuming"
        st.metric("⚡ Power", f"{abs(state['bms']['power']):.1f} kW",
                    delta=power_label)


    with col3:
        st.metric("🏎️ Speed", f"{state['dynamics']['speed_kmh']:.1f} km/h",
                    delta=f"{state['dynamics']['acceleration_mps2']:.2f} m/s²")


    with col4:
        temp_status = "Cooling" if state['bms']['cooling_active'] else "Heating" if state['bms']['heating_active'] else "Normal"
        st.metric("🌡️ Battery Temp", f"{state['bms']['temperature']:.1f}°C",
                    delta=temp_status)


    with col5:
        st.metric("📍 Range", f"{state['estimated_range_km']:.0f} km",
                    delta=f"{state['dynamics']['distance_km']:.1f} km driven")


    # Alerts
    if state['bms']['errors']:
        for error in state['bms']['errors']:
            st.error(f"🚨 {error}")


    if state['bms']['warnings']:
        for warning in state['bms']['warnings']:
            st.warning(f"⚠️ {warning}")


def main():
    """Main application"""
    initialize_session_state()
    ev = st.session_state.ev_twin


    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚡ Tata Nexon EV Digital Twin - RT Operation")
        st.caption("Real-time continuous monitoring | 40.5 kWh Battery | 312 km Range")
    with col2:
        st.metric("Vehicle ID", ev.vehicle_id)


    # Sidebar
    st.sidebar.header("🎛️ Vehicle Controls")


    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("▶️ Start" if not st.session_state.simulation_running else "⏸️ Pause", 
                                use_container_width=True, key="start_pause"):
            st.session_state.simulation_running = not st.session_state.simulation_running
            st.rerun()


    with col2:
        if st.sidebar.button("🔄 Reset", use_container_width=True, key="reset"):
            st.session_state.ev_twin = TataNexonEVDigitalTwin()
            st.session_state.simulation_running = False
            initialize_session_state()
            st.rerun()


    st.sidebar.markdown("---")


    # Driving mode
    st.sidebar.subheader("🚗 Driving Mode")
    mode_selection = st.sidebar.selectbox(
        "Select Mode",
        [m.value for m in DrivingMode],
        key="driving_mode"
    )


    if mode_selection == DrivingMode.ECO.value:
        ev.driving_mode = DrivingMode.ECO
        ev.powertrain.set_driving_mode(DrivingMode.ECO)
        st.sidebar.info("🌱 Eco Mode: Max efficiency")
    elif mode_selection == DrivingMode.SPORT.value:
        ev.driving_mode = DrivingMode.SPORT
        ev.powertrain.set_driving_mode(DrivingMode.SPORT)
        st.sidebar.success("🏁 Sport Mode: Full power")
    else:
        ev.driving_mode = DrivingMode.CITY
        ev.powertrain.set_driving_mode(DrivingMode.CITY)
        st.sidebar.info("🏙️ City Mode: Balanced")


    st.sidebar.markdown("---")


    # Driver inputs
    st.sidebar.subheader("🎮 Driver Inputs")
    ev.accelerator_position = st.sidebar.slider("Accelerator (%)", 0, 100, 0, 5, key="accel")
    ev.brake_position = st.sidebar.slider("Brake Pedal (%)", 0, 100, 0, 5, key="brake")


    # Regen level
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔋 Regenerative Braking")
    regen_labels = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}
    ev.braking.regen_level = st.sidebar.select_slider(
        "Regen Level",
        options=[0, 1, 2, 3],
        value=2,
        format_func=lambda x: regen_labels[x],
        key="regen"
    )


    st.sidebar.markdown("---")


    # Environment
    st.sidebar.subheader("🌡️ Environment")
    ev.ambient_temperature = st.sidebar.slider("Ambient Temp (°C)", 0, 50, 30, 1, key="temp")


    st.sidebar.markdown("---")


    # Charging
    st.sidebar.subheader("⚡ Charging")
    protocol = st.sidebar.selectbox(
        "Charging Type",
        [p.value for p in ChargingProtocol],
        key="protocol"
    )


    if st.sidebar.button("🔌 Start Charging", key="start_charge"):
        result = ev.charging.initiate_charging(protocol, 100, ev.bms.soc, ev.config['capacity_kwh'])
        st.sidebar.success(f"Charging: {result['power_kw']} kW")


    if st.sidebar.button("⏹️ Stop Charging", key="stop_charge"):
        ev.charging.stop_charging()
        st.sidebar.info("Charging stopped")


    # Real-time KPIs (auto-updates every 0.5s)
    render_realtime_kpis()


    st.markdown("---")


    # Get current state for tabs
    state = st.session_state.current_state


    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔋 Battery", "⚡ Powertrain", "🛑 Braking", "🔌 Charging", "📈 Analytics"
    ])


    with tab1:
        st.subheader("Battery Management System")


        col1, col2, col3 = st.columns(3)


        with col1:
            st.markdown("#### Battery Status")
            df = pd.DataFrame({
                "Parameter": ["Voltage", "Current", "Temperature", "Power", "Coolant Temp"],
                "Value": [
                    f"{state['bms']['voltage']:.1f} V",
                    f"{state['bms']['current']:.1f} A",
                    f"{state['bms']['temperature']:.1f} °C",
                    f"{state['bms']['power']:.1f} kW",
                    f"{state['bms']['coolant_temp']:.1f} °C"
                ]
            })
            st.dataframe(df, hide_index=True, use_container_width=True)


        with col2:
            st.markdown("#### Thermal System")
            thermal_df = pd.DataFrame({
                "System": ["Liquid Cooling", "Battery Heating", "Temp Limit", "Status"],
                "State": [
                    "🟢 Active" if state['bms']['cooling_active'] else "⚪ Standby",
                    "🟢 Active" if state['bms']['heating_active'] else "⚪ Standby",
                    "Max 60°C",
                    "🟢 Normal" if 15 <= state['bms']['temperature'] <= 50 else "⚠️ Sub-optimal"
                ]
            })
            st.dataframe(thermal_df, hide_index=True, use_container_width=True)


        with col3:
            st.markdown("#### Specifications")
            st.info(f"""
            **Capacity:** {ev.config['capacity_kwh']} kWh
            **Voltage:** {ev.config['nominal_voltage']}V
            **Cooling:** Liquid cooling
            **Max Temp:** 60°C
            """)


        # SOC-OCV Curve
        soc_curve = np.linspace(0, 100, 100)
        ocv_curve = [ev.bms._soc_to_ocv(s) for s in soc_curve]


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=soc_curve, y=ocv_curve, mode='lines', name='OCV Curve'))
        fig.add_trace(go.Scatter(x=[state['bms']['soc']], y=[state['bms']['voltage']], 
                                mode='markers', marker=dict(size=15, color='red'), name='Current'))
        fig.update_layout(xaxis_title="SOC (%)", yaxis_title="Voltage (V)",
                            height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


    with tab2:
        st.subheader("Powertrain System")


        col1, col2 = st.columns(2)


        with col1:
            st.markdown(f"#### Motor ({state['powertrain']['driving_mode']})")
            motor_df = pd.DataFrame({
                "Parameter": ["RPM", "Torque", "Power", "Max Power"],
                "Value": [
                    f"{state['powertrain']['motor_rpm']:.0f}",
                    f"{state['powertrain']['motor_torque']:.1f} Nm",
                    f"{state['powertrain']['power_kw']:.1f} kW",
                    f"{ev.config['max_power_kw']} kW"
                ]
            })
            st.dataframe(motor_df, hide_index=True, use_container_width=True)


        with col2:
            st.markdown("#### Vehicle Dynamics")
            dyn_df = pd.DataFrame({
                "Parameter": ["Speed", "Acceleration", "Distance"],
                "Value": [
                    f"{state['dynamics']['speed_kmh']:.1f} km/h",
                    f"{state['dynamics']['acceleration_mps2']:.2f} m/s²",
                    f"{state['dynamics']['distance_km']:.2f} km"
                ]
            })
            st.dataframe(dyn_df, hide_index=True, use_container_width=True)


    with tab3:
        st.subheader("Braking System")


        col1, col2 = st.columns(2)


        with col1:
            st.markdown("#### Braking Status")
            brake_df = pd.DataFrame({
                "Parameter": ["Pressure", "Regen Braking", "Mechanical", "ABS", "Regen Power"],
                "Value": [
                    f"{state['braking']['brake_pressure']:.0f}%",
                    "🟢 Active" if state['braking']['regen_active'] else "⚪ Off",
                    "🟢 Active" if state['braking']['mechanical_active'] else "⚪ Off",
                    "🟢 Active" if state['braking']['abs_active'] else "⚪ Standby",
                    f"{abs(state['braking']['regen_power_kw']):.1f} kW"
                ]
            })
            st.dataframe(brake_df, hide_index=True, use_container_width=True)


        with col2:
            st.markdown("#### Energy Recovery")
            recovery_df = pd.DataFrame({
                "Metric": ["Recovered", "Front Brake Temp", "Rear Brake Temp"],
                "Value": [
                    f"{state['braking']['energy_recovered']:.2f} kWh",
                    f"{state['braking']['brake_temp_front']:.1f}°C",
                    f"{state['braking']['brake_temp_rear']:.1f}°C"
                ]
            })
            st.dataframe(recovery_df, hide_index=True, use_container_width=True)


    with tab4:
        st.subheader("Charging System")


        if ev.charging.charging_active:
            st.success("🔌 Charging Active")
            charging_state = ev.charging.update_charging(ev.bms.soc, 100, ev.bms.temperature)
            if charging_state['charging_active']:
                charge_df = pd.DataFrame({
                    "Parameter": ["Protocol", "Power", "SOC"],
                    "Value": [
                        ev.charging.charging_protocol,
                        f"{charging_state['actual_power_kw']:.1f} kW",
                        f"{ev.bms.soc:.1f}%"
                    ]
                })
                st.dataframe(charge_df, hide_index=True, use_container_width=True)
        else:
            st.info("⚪ Not Charging")


        charge_options = pd.DataFrame({
            "Type": ["DC Fast", "AC Fast", "Home"],
            "Power": ["50 kW", "7.2 kW", "3.3 kW"],
            "0-80%": ["~50 min", "~6 hrs", "~12 hrs"]
        })
        st.dataframe(charge_options, hide_index=True, use_container_width=True)


    with tab5:
        st.subheader("Vehicle Analytics")


        if len(ev.history['soc']) > 5:
            plot_vars = st.multiselect(
                "Select metrics",
                ['SOC (%)', 'Speed (km/h)', 'Power (kW)', 'Temperature (°C)'],
                default=['SOC (%)', 'Power (kW)'],
                key="metrics"
            )


            if plot_vars:
                fig = go.Figure()
                data_map = {
                    'SOC (%)': ev.history['soc'][-100:],
                    'Speed (km/h)': ev.history['speed'][-100:],
                    'Power (kW)': ev.history['power'][-100:],
                    'Temperature (°C)': ev.history['temperature'][-100:]
                }


                for var in plot_vars:
                    fig.add_trace(go.Scatter(y=data_map[var], mode='lines', name=var))


                fig.update_layout(xaxis_title="Time", yaxis_title="Value",
                                height=350, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)


            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Power", f"{np.mean([abs(p) for p in ev.history['power']]):.1f} kW")
            with col2:
                st.metric("Max Speed", f"{np.max(ev.history['speed']):.1f} km/h")
            with col3:
                st.metric("Energy Used", f"{ev.bms.energy_consumed:.2f} kWh")
            with col4:
                st.metric("Recovered", f"{ev.braking.energy_recovered_kwh:.2f} kWh")


            if st.button("📥 Download CSV", key="download"):
                df_export = pd.DataFrame({
                    'SOC': ev.history['soc'],
                    'Speed': ev.history['speed'],
                    'Power': ev.history['power'],
                    'Temp': ev.history['temperature']
                })
                csv = df_export.to_csv(index=False)
                st.download_button("Download", csv, 
                                f"nexon_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv", key="dl_btn")
        else:
            st.info("📊 Start simulation to collect data")


    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #0D47A1;'>
        <p><strong>Tata Nexon EV Digital Twin - RT Operation</strong></p>
        <p style='font-size: 0.9em;'>⚡ Real-time auto-refresh every 0.5 seconds | Max Battery Temp: 60°C</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
