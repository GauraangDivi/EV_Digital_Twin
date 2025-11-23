# ev_digital_twin_realtime.py
"""
EV Digital Twin - OEM/Manufacturer Dashboard
Real-time Streamlit application with smooth live updates
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from enum import Enum

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="EV Digital Twin - Real-time Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional OEM dashboard
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== PROTOCOLS AND STANDARDS ====================

class CANProtocol(Enum):
    """CAN Bus Protocol - ISO 15765-4"""
    BITRATE_250K = 250000
    BITRATE_500K = 500000
    FRAME_LENGTH = 8

class ChargingProtocol(Enum):
    """EV Charging Communication Protocols"""
    ISO_15118 = "ISO 15118 (V2G)"
    CCS_COMBO = "CCS Combo (250kW)"
    CHADEMO = "CHAdeMO"
    IEC_61851 = "IEC 61851 (AC L2)"

# ==================== BATTERY MANAGEMENT SYSTEM ====================

@dataclass
class BatteryCell:
    """Individual battery cell parameters"""
    voltage: float = 3.7
    temperature: float = 25.0
    capacity_ah: float = 4.8
    internal_resistance: float = 0.02
    cycle_count: int = 0

class BatteryManagementSystem:
    """Advanced BMS with SOC/SOH estimation and thermal management"""

    def __init__(self, config):
        self.config = config
        self.soc = 85.0
        self.soh = 100.0
        self.voltage = 370.0
        self.current = 0.0
        self.temperature = 25.0
        self.power = 0.0
        self.energy_consumed = 0.0
        self.cycle_count = 0

        # Kalman filter parameters
        self.kf_estimate = self.soc
        self.kf_error_covariance = 1.0
        self.process_noise = 0.01
        self.measurement_noise = 0.1

    def _soc_to_ocv(self, soc):
        """OCV lookup table for Li-ion NMC chemistry"""
        soc_points = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ocv_points = np.array([300, 330, 345, 355, 365, 370, 375, 385, 395, 410, 420])
        return np.interp(soc, soc_points, ocv_points)

    def estimate_soc_coulomb_counting(self, current_a, dt_hours):
        """SOC estimation using Coulomb Counting"""
        capacity_ah = self.config['capacity_kwh'] * 1000 / self.config['nominal_voltage']
        delta_soc = (current_a * dt_hours / capacity_ah) * 100
        self.soc = np.clip(self.soc - delta_soc, 0, 100)
        return self.soc

    def estimate_soc_kalman_filter(self, measured_voltage, current_a):
        """Kalman Filter SOC estimation"""
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
        """SOH estimation based on capacity fade and cycle count"""
        capacity_fade = 100 * np.exp(-0.00005 * self.cycle_count)
        resistance_growth = 100 * (0.02 / (0.02 + self.cycle_count * 0.000001))
        self.soh = (0.7 * capacity_fade + 0.3 * resistance_growth)
        self.soh = np.clip(self.soh, 50, 100)
        return self.soh

    def thermal_management(self, ambient_temp):
        """Thermal management with active cooling"""
        heat_generation = (self.current ** 2) * 0.02
        cooling_coefficient = 0.5 if self.temperature > 35 else 0.2
        heat_dissipation = cooling_coefficient * (self.temperature - ambient_temp)

        thermal_mass = 50
        dt = 1
        temp_change = (heat_generation - heat_dissipation) * dt / thermal_mass
        self.temperature = self.temperature + temp_change

        cooling_active = self.temperature > 35
        return self.temperature, cooling_active

    def safety_check(self):
        """Safety monitoring per ISO 26262"""
        warnings = []
        errors = []

        if self.voltage > self.config['max_voltage'] * 1.05:
            errors.append("CRITICAL: Overvoltage detected")
        elif self.voltage > self.config['max_voltage']:
            warnings.append("Warning: Approaching voltage limit")

        if self.temperature > 55:
            errors.append("CRITICAL: Overtemperature - Thermal runaway risk")
        elif self.temperature > 45:
            warnings.append("Warning: High temperature")

        if self.soc < 5:
            errors.append("CRITICAL: Battery critically low")
        elif self.soc < 15:
            warnings.append("Warning: Low battery")

        return warnings, errors

    def update(self, current_a, ambient_temp, dt_seconds=1.0):
        """Main BMS update cycle"""
        dt_hours = dt_seconds / 3600.0

        self.current = current_a
        self.voltage = self._soc_to_ocv(self.soc)
        self.power = (self.voltage * self.current) / 1000.0

        # Hybrid SOC estimation
        soc_coulomb = self.estimate_soc_coulomb_counting(current_a, dt_hours)
        soc_kalman = self.estimate_soc_kalman_filter(self.voltage, current_a)
        self.soc = 0.7 * soc_kalman + 0.3 * soc_coulomb

        # SOH estimation
        if abs(current_a) > 10:
            self.cycle_count += dt_hours * 0.0001
        self.estimate_soh()

        # Thermal management
        self.temperature, cooling_active = self.thermal_management(ambient_temp)

        # Safety checks
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
            'warnings': warnings,
            'errors': errors
        }

# ==================== POWERTRAIN SYSTEM ====================

class PowertrainSystem:
    """Electric motor and inverter control with regenerative braking"""

    def __init__(self, config):
        self.config = config
        self.motor_rpm_front = 0
        self.motor_rpm_rear = 0
        self.motor_torque_front = 0
        self.motor_torque_rear = 0
        self.motor_efficiency = 0.95
        self.vehicle_speed_kmh = 0
        self.regen_active = False

    def calculate_motor_speed(self, vehicle_speed_kmh):
        """Calculate motor RPM from vehicle speed"""
        wheel_diameter_m = 0.705
        gear_ratio = 9.0
        wheel_rpm = (vehicle_speed_kmh * 1000 / 60) / (np.pi * wheel_diameter_m)
        motor_rpm = wheel_rpm * gear_ratio
        return motor_rpm

    def calculate_torque_demand(self, accelerator_pct, brake_pct, vehicle_speed_kmh):
        """Calculate torque with AWD torque vectoring"""
        max_torque = self.config['max_torque_nm']

        if accelerator_pct > 0:
            base_torque = (accelerator_pct / 100) * max_torque

            # Torque derating at high speed
            if vehicle_speed_kmh > 100:
                derating_factor = 1.0 - (vehicle_speed_kmh - 100) / 200
                base_torque *= max(0.3, derating_factor)

            # 40% front, 60% rear split
            self.motor_torque_front = base_torque * 0.4
            self.motor_torque_rear = base_torque * 0.6
            self.regen_active = False

        elif brake_pct > 0:
            regen_torque = -(brake_pct / 100) * max_torque * 0.6
            self.motor_torque_front = regen_torque * 0.4
            self.motor_torque_rear = regen_torque * 0.6
            self.regen_active = True
        else:
            self.motor_torque_front = 0
            self.motor_torque_rear = 0
            self.regen_active = False

        return self.motor_torque_front + self.motor_torque_rear

    def calculate_power(self, total_torque_nm, motor_rpm):
        """Calculate mechanical power"""
        power_w = (total_torque_nm * motor_rpm * 2 * np.pi) / 60
        power_kw = power_w / 1000

        if total_torque_nm > 0:
            power_kw = power_kw / (self.motor_efficiency * 0.97)
        else:
            power_kw = power_kw * self.motor_efficiency * 0.97

        return power_kw

    def update(self, accelerator_pct, brake_pct, vehicle_speed_kmh):
        """Update powertrain state"""
        self.vehicle_speed_kmh = vehicle_speed_kmh
        motor_rpm = self.calculate_motor_speed(vehicle_speed_kmh)
        self.motor_rpm_front = motor_rpm
        self.motor_rpm_rear = motor_rpm

        total_torque = self.calculate_torque_demand(accelerator_pct, brake_pct, vehicle_speed_kmh)
        power_kw = self.calculate_power(total_torque, motor_rpm)

        return {
            'motor_rpm_front': self.motor_rpm_front,
            'motor_rpm_rear': self.motor_rpm_rear,
            'motor_torque_front': self.motor_torque_front,
            'motor_torque_rear': self.motor_torque_rear,
            'total_torque': total_torque,
            'power_kw': power_kw,
            'regen_active': self.regen_active
        }

# ==================== CHARGING SYSTEM ====================

class ChargingSystem:
    """EV Charging with ISO 15118 and CCS support"""

    def __init__(self):
        self.charging_active = False
        self.charging_protocol = None
        self.charging_power_kw = 0
        self.charging_current_a = 0
        self.charge_time_remaining_min = 0

    def initiate_charging(self, protocol, target_soc, current_soc, battery_capacity_kwh):
        """Initiate charging session"""
        self.charging_protocol = protocol
        self.charging_active = True

        if "CCS" in protocol or "250kW" in protocol:
            self.charging_power_kw = 250
            self.charging_current_a = 625
        elif "CHAdeMO" in protocol:
            self.charging_power_kw = 62.5
            self.charging_current_a = 125
        else:
            self.charging_power_kw = 11
            self.charging_current_a = 48

        energy_needed = (target_soc - current_soc) / 100 * battery_capacity_kwh
        self.charge_time_remaining_min = (energy_needed / self.charging_power_kw) * 60

        return {
            'status': 'Charging initiated',
            'protocol': self.charging_protocol,
            'power_kw': self.charging_power_kw,
            'time_remaining_min': self.charge_time_remaining_min
        }

    def update_charging(self, current_soc, target_soc, dt_seconds=1.0):
        """Update charging with power tapering"""
        if not self.charging_active:
            return {'charging_active': False}

        # Power tapering above 80% SOC
        if current_soc > 80:
            taper_factor = 1.0 - ((current_soc - 80) / 20) * 0.7
            actual_power = self.charging_power_kw * taper_factor
        else:
            actual_power = self.charging_power_kw

        if current_soc >= target_soc:
            self.stop_charging()
            return {'charging_active': False, 'status': 'Charging complete'}

        return {
            'charging_active': True,
            'actual_power_kw': actual_power,
            'time_remaining_min': self.charge_time_remaining_min
        }

    def stop_charging(self):
        """Stop charging session"""
        self.charging_active = False
        self.charging_power_kw = 0

# ==================== VEHICLE DYNAMICS ====================

class VehicleDynamics:
    """Vehicle physics simulation"""

    def __init__(self, config):
        self.config = config
        self.speed_kmh = 0
        self.acceleration_mps2 = 0
        self.distance_km = 0

    def update(self, motor_torque_nm, dt_seconds=1.0):
        """Update vehicle dynamics"""
        mass_kg = 1847
        wheel_radius_m = 0.3525
        gear_ratio = 9.0
        air_drag_coeff = 0.23
        frontal_area_m2 = 2.22
        rolling_resistance_coeff = 0.01

        wheel_force_n = (motor_torque_nm * gear_ratio) / wheel_radius_m
        drag_force_n = 0.5 * 1.225 * air_drag_coeff * frontal_area_m2 * (self.speed_kmh / 3.6) ** 2
        rolling_resistance_n = rolling_resistance_coeff * mass_kg * 9.81

        net_force_n = wheel_force_n - drag_force_n - rolling_resistance_n
        self.acceleration_mps2 = net_force_n / mass_kg

        delta_speed_mps = self.acceleration_mps2 * dt_seconds
        self.speed_kmh += delta_speed_mps * 3.6
        self.speed_kmh = max(0, self.speed_kmh)

        self.distance_km += (self.speed_kmh / 3600) * dt_seconds

        return {
            'speed_kmh': self.speed_kmh,
            'acceleration_mps2': self.acceleration_mps2,
            'distance_km': self.distance_km
        }

# ==================== DIGITAL TWIN ====================

class EVDigitalTwin:
    """Complete EV Digital Twin - Tesla Model 3 Long Range"""

    def __init__(self, vehicle_id="DEMO-001"):
        self.vehicle_id = vehicle_id
        self.config = {
            'model': 'Tesla Model 3 Long Range',
            'capacity_kwh': 82,
            'nominal_voltage': 355,
            'max_voltage': 403,
            'min_voltage': 260,
            'max_discharge_current': 500,
            'max_torque_nm': 639,
            'range_wltp_km': 629,
        }

        self.bms = BatteryManagementSystem(self.config)
        self.powertrain = PowertrainSystem(self.config)
        self.charging = ChargingSystem()
        self.dynamics = VehicleDynamics(self.config)

        self.ambient_temperature = 25.0
        self.accelerator_position = 0
        self.brake_position = 0

        self.history = {
            'time': [],
            'soc': [],
            'soh': [],
            'speed': [],
            'power': [],
            'temperature': [],
        }

    def update(self, dt_seconds=0.5):
        """Main update loop"""
        # Update powertrain
        powertrain_state = self.powertrain.update(
            self.accelerator_position,
            self.brake_position,
            self.dynamics.speed_kmh
        )

        # Calculate battery current
        if powertrain_state['power_kw'] != 0:
            battery_current = (powertrain_state['power_kw'] * 1000) / max(self.bms.voltage, 1)
        else:
            battery_current = 0

        # Charging override
        if self.charging.charging_active:
            charging_state = self.charging.update_charging(self.bms.soc, 100, dt_seconds)
            if charging_state['charging_active']:
                battery_current = -charging_state['actual_power_kw'] * 1000 / max(self.bms.voltage, 1)

        # Update BMS
        bms_state = self.bms.update(battery_current, self.ambient_temperature, dt_seconds)

        # Update dynamics
        dynamics_state = self.dynamics.update(powertrain_state['total_torque'], dt_seconds)

        # Log history (limit to last 500 points)
        self.history['time'].append(datetime.now())
        self.history['soc'].append(bms_state['soc'])
        self.history['soh'].append(bms_state['soh'])
        self.history['speed'].append(dynamics_state['speed_kmh'])
        self.history['power'].append(bms_state['power'])
        self.history['temperature'].append(bms_state['temperature'])

        # Keep only last 500 data points to prevent memory issues
        max_history = 500
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]

        # Calculate range
        remaining_energy = (bms_state['soc'] / 100) * self.config['capacity_kwh']
        estimated_range = remaining_energy / 0.15

        return {
            'bms': bms_state,
            'powertrain': powertrain_state,
            'dynamics': dynamics_state,
            'estimated_range_km': estimated_range
        }

# ==================== STREAMLIT UI ====================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'ev_twin' not in st.session_state:
        st.session_state.ev_twin = EVDigitalTwin()
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.title("🚗 EV Digital Twin - Real-time Dashboard")
    with col3:
        st.metric("Vehicle ID", st.session_state.ev_twin.vehicle_id)

def render_sidebar_controls(ev):
    """Render sidebar with vehicle controls"""
    st.sidebar.header("🎛️ Vehicle Controls")

    # Simulation control
    sim_col1, sim_col2 = st.sidebar.columns(2)
    with sim_col1:
        if st.sidebar.button("▶️ Start" if not st.session_state.simulation_running else "⏸️ Pause", 
                     use_container_width=True):
            st.session_state.simulation_running = not st.session_state.simulation_running
            st.rerun()

    with sim_col2:
        if st.sidebar.button("🔄 Reset", use_container_width=True):
            st.session_state.ev_twin = EVDigitalTwin()
            st.session_state.simulation_running = False
            st.rerun()

    st.sidebar.markdown("---")

    # Quick scenarios
    st.sidebar.subheader("📋 Quick Scenarios")
    scenario = st.sidebar.selectbox(
        "Load Scenario",
        ["Manual Control", "City Driving", "Highway Cruise", "Aggressive Acceleration", "Charging Session"],
        key="scenario_selector"
    )

    if scenario == "City Driving":
        ev.accelerator_position = 30
        ev.brake_position = 0
    elif scenario == "Highway Cruise":
        ev.accelerator_position = 50
        ev.brake_position = 0
    elif scenario == "Aggressive Acceleration":
        ev.accelerator_position = 100
        ev.brake_position = 0
    elif scenario == "Charging Session":
        if st.sidebar.button("Start Fast Charging"):
            ev.charging.initiate_charging(ChargingProtocol.CCS_COMBO.value, 100, ev.bms.soc, ev.config['capacity_kwh'])
            st.sidebar.success("Fast charging initiated!")
    else:
        # Manual controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚗 Driver Inputs")
        ev.accelerator_position = st.sidebar.slider("Accelerator (%)", 0, 100, ev.accelerator_position, 5, key="accel_slider")
        ev.brake_position = st.sidebar.slider("Brake (%)", 0, 100, ev.brake_position, 5, key="brake_slider")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡️ Environment")
    ev.ambient_temperature = st.sidebar.slider("Ambient Temp (°C)", -20, 45, int(ev.ambient_temperature), 1, key="temp_slider")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Charging")
    protocol = st.sidebar.selectbox(
        "Protocol",
        [p.value for p in ChargingProtocol],
        key="protocol_selector"
    )
    target_soc = st.sidebar.slider("Target SOC (%)", 0, 100, 100, 5, key="target_soc_slider")

    if st.sidebar.button("🔌 Start Charging"):
        result = ev.charging.initiate_charging(protocol, target_soc, ev.bms.soc, ev.config['capacity_kwh'])
        st.sidebar.success(f"Charging: {result['power_kw']} kW")

    if st.sidebar.button("⏹️ Stop Charging"):
        ev.charging.stop_charging()
        st.sidebar.info("Charging stopped")

@st.fragment(run_every="0.5s")
def render_realtime_metrics():
    """Render real-time updating metrics"""
    ev = st.session_state.ev_twin

    # Update simulation if running
    if st.session_state.simulation_running:
        current_time = time.time()
        dt = current_time - st.session_state.last_update
        state = ev.update(dt_seconds=min(dt, 1.0))
        st.session_state.last_update = current_time
    else:
        # Static state when paused
        state = {
            'bms': {
                'soc': ev.bms.soc,
                'soh': ev.bms.soh,
                'voltage': ev.bms.voltage,
                'current': ev.bms.current,
                'temperature': ev.bms.temperature,
                'power': ev.bms.power,
                'cooling_active': False,
                'warnings': [],
                'errors': []
            },
            'powertrain': {
                'motor_rpm_front': 0,
                'motor_rpm_rear': 0,
                'motor_torque_front': 0,
                'motor_torque_rear': 0,
                'total_torque': 0,
                'power_kw': 0,
                'regen_active': False
            },
            'dynamics': {
                'speed_kmh': ev.dynamics.speed_kmh,
                'acceleration_mps2': 0,
                'distance_km': ev.dynamics.distance_km
            },
            'estimated_range_km': (ev.bms.soc / 100) * ev.config['range_wltp_km']
        }

    # Render KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "🔋 Battery SOC",
            f"{state['bms']['soc']:.1f}%",
            delta=f"SOH: {state['bms']['soh']:.1f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "⚡ Power",
            f"{state['bms']['power']:.1f} kW",
            delta="Regen" if state['powertrain']['regen_active'] else "Drive",
            delta_color="inverse" if state['powertrain']['regen_active'] else "normal"
        )

    with col3:
        st.metric(
            "🏎️ Speed",
            f"{state['dynamics']['speed_kmh']:.1f} km/h",
            delta=f"{state['dynamics']['acceleration_mps2']:.2f} m/s²"
        )

    with col4:
        st.metric(
            "🌡️ Battery Temp",
            f"{state['bms']['temperature']:.1f}°C",
            delta="Cooling ON" if state['bms']['cooling_active'] else "Normal",
            delta_color="inverse" if state['bms']['cooling_active'] else "normal"
        )

    with col5:
        st.metric(
            "📍 Range",
            f"{state['estimated_range_km']:.0f} km",
            delta=f"{state['dynamics']['distance_km']:.1f} km driven"
        )

    # Safety alerts
    if state['bms']['errors']:
        for error in state['bms']['errors']:
            st.error(f"🚨 {error}")

    if state['bms']['warnings']:
        for warning in state['bms']['warnings']:
            st.warning(f"⚠️ {warning}")

    # Return state for use in tabs
    return state

def render_battery_tab(ev, state):
    """Render battery management tab"""
    st.subheader("🔋 Battery Management System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Battery Status")
        status_data = {
            "Parameter": ["Voltage", "Current", "Temperature", "Power", "Energy Used", "Cycle Count"],
            "Value": [
                f"{state['bms']['voltage']:.1f} V",
                f"{state['bms']['current']:.1f} A",
                f"{state['bms']['temperature']:.1f} °C",
                f"{state['bms']['power']:.1f} kW",
                f"{ev.bms.energy_consumed:.2f} kWh",
                f"{ev.bms.cycle_count:.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(status_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### System Health")
        health_data = {
            "Component": ["Overall SOH", "Cooling System", "Cell Balancing", "Safety Status"],
            "Status": [
                f"{state['bms']['soh']:.1f}%",
                "🟢 Active" if state['bms']['cooling_active'] else "⚪ Inactive",
                "🟢 Balanced",
                "🟢 OK" if not state['bms']['errors'] else "🔴 ERROR"
            ]
        }
        st.dataframe(pd.DataFrame(health_data), hide_index=True, use_container_width=True)

    with col3:
        st.markdown("#### Protocols")
        st.info("""
        **Active Standards:**
        - ISO 26262 (Safety)
        - Kalman Filter SOC
        - Coulomb Counting
        - Active Thermal Mgmt
        """)

    # SOC-OCV Curve
    st.markdown("#### State of Charge vs Open Circuit Voltage")
    soc_curve = np.linspace(0, 100, 100)
    ocv_curve = [ev.bms._soc_to_ocv(s) for s in soc_curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=soc_curve, y=ocv_curve, mode='lines', name='OCV Curve', line=dict(color='blue')))
    fig.add_trace(go.Scatter(
        x=[state['bms']['soc']], 
        y=[state['bms']['voltage']], 
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='Current State'
    ))
    fig.update_layout(
        xaxis_title="State of Charge (%)",
        yaxis_title="Open Circuit Voltage (V)",
        height=350,
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_powertrain_tab(state):
    """Render powertrain tab"""
    st.subheader("⚡ Powertrain System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Motor Status")
        motor_data = {
            "Parameter": [
                "Front Motor RPM",
                "Rear Motor RPM",
                "Front Torque",
                "Rear Torque",
                "Total Torque",
                "Motor Power"
            ],
            "Value": [
                f"{state['powertrain']['motor_rpm_front']:.0f}",
                f"{state['powertrain']['motor_rpm_rear']:.0f}",
                f"{state['powertrain']['motor_torque_front']:.1f} Nm",
                f"{state['powertrain']['motor_torque_rear']:.1f} Nm",
                f"{state['powertrain']['total_torque']:.1f} Nm",
                f"{state['powertrain']['power_kw']:.1f} kW"
            ]
        }
        st.dataframe(pd.DataFrame(motor_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Vehicle Dynamics")
        dynamics_data = {
            "Parameter": [
                "Vehicle Speed",
                "Acceleration",
                "Distance Traveled",
                "Drive Mode",
                "Regen Braking",
                "Efficiency"
            ],
            "Value": [
                f"{state['dynamics']['speed_kmh']:.1f} km/h",
                f"{state['dynamics']['acceleration_mps2']:.2f} m/s²",
                f"{state['dynamics']['distance_km']:.2f} km",
                "AWD Dual Motor",
                "🟢 Active" if state['powertrain']['regen_active'] else "⚪ Inactive",
                "95% (Motor) × 97% (Inverter)"
            ]
        }
        st.dataframe(pd.DataFrame(dynamics_data), hide_index=True, use_container_width=True)

    # Torque vectoring visualization
    st.markdown("#### Torque Distribution (AWD)")
    fig = go.Figure(data=[
        go.Bar(
            x=['Front Motor', 'Rear Motor'],
            y=[abs(state['powertrain']['motor_torque_front']), abs(state['powertrain']['motor_torque_rear'])],
            marker_color=['#3B82F6', '#10B981'],
            text=[f"{abs(state['powertrain']['motor_torque_front']):.0f} Nm", 
                  f"{abs(state['powertrain']['motor_torque_rear']):.0f} Nm"],
            textposition='auto',
        )
    ])
    fig.update_layout(
        yaxis_title="Torque (Nm)",
        height=300,
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def render_charging_tab(ev):
    """Render charging tab"""
    st.subheader("🔌 Charging System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Charging Status")
        if ev.charging.charging_active:
            st.success("🔌 Charging Active")
            charging_state = ev.charging.update_charging(ev.bms.soc, 100)
            if charging_state['charging_active']:
                charge_info = {
                    "Parameter": [
                        "Protocol",
                        "Charging Power",
                        "Charging Current",
                        "Time Remaining",
                        "Current SOC"
                    ],
                    "Value": [
                        ev.charging.charging_protocol,
                        f"{charging_state['actual_power_kw']:.1f} kW",
                        f"{ev.charging.charging_current_a:.1f} A",
                        f"{charging_state['time_remaining_min']:.1f} min",
                        f"{ev.bms.soc:.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(charge_info), hide_index=True, use_container_width=True)
        else:
            st.info("⚪ Not Charging")
            st.write("Use the sidebar to initiate charging session")

    with col2:
        st.markdown("#### Supported Protocols")
        protocols_data = {
            "Protocol": [
                "ISO 15118",
                "CCS Combo",
                "CHAdeMO",
                "IEC 61851"
            ],
            "Type": [
                "V2G, Plug & Charge",
                "DC Fast (250 kW)",
                "DC Fast (62.5 kW)",
                "AC Level 2 (11 kW)"
            ],
            "Status": [
                "✅ Supported",
                "✅ Supported",
                "✅ Supported",
                "✅ Supported"
            ]
        }
        st.dataframe(pd.DataFrame(protocols_data), hide_index=True, use_container_width=True)

    # Charging curve visualization
    if len(ev.history['soc']) > 5:
        st.markdown("#### Charging Curve (SOC vs Time)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=ev.history['soc'][-100:],
            mode='lines',
            name='SOC',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            xaxis_title="Time (iterations)",
            yaxis_title="State of Charge (%)",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_analytics_tab(ev):
    """Render analytics and history tab"""
    st.subheader("📈 Vehicle Analytics")

    if len(ev.history['soc']) > 5:
        # Plot selection
        col1, col2 = st.columns([3, 1])
        with col1:
            plot_vars = st.multiselect(
                "Select metrics to visualize",
                options=['SOC (%)', 'SOH (%)', 'Speed (km/h)', 'Power (kW)', 'Temperature (°C)'],
                default=['SOC (%)', 'Power (kW)', 'Speed (km/h)'],
                key="plot_selector"
            )

        with col2:
            show_points = st.toggle("Show data points", value=False, key="show_points_toggle")

        if plot_vars:
            fig = go.Figure()

            data_mapping = {
                'SOC (%)': ev.history['soc'][-100:],
                'SOH (%)': ev.history['soh'][-100:],
                'Speed (km/h)': ev.history['speed'][-100:],
                'Power (kW)': ev.history['power'][-100:],
                'Temperature (°C)': ev.history['temperature'][-100:]
            }

            for var in plot_vars:
                mode = 'lines+markers' if show_points else 'lines'
                fig.add_trace(go.Scatter(
                    y=data_mapping[var],
                    mode=mode,
                    name=var,
                    line=dict(width=2)
                ))

            fig.update_layout(
                xaxis_title="Time (iterations)",
                yaxis_title="Value",
                height=400,
                template="plotly_white",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("#### Performance Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Avg Power", f"{np.mean(ev.history['power']):.1f} kW")
        with col2:
            st.metric("Max Speed", f"{np.max(ev.history['speed']):.1f} km/h")
        with col3:
            st.metric("Avg Temp", f"{np.mean(ev.history['temperature']):.1f}°C")
        with col4:
            st.metric("Energy Used", f"{ev.bms.energy_consumed:.2f} kWh")
        with col5:
            efficiency = (ev.dynamics.distance_km / max(ev.bms.energy_consumed, 0.001)) if ev.bms.energy_consumed > 0 else 0
            st.metric("Efficiency", f"{efficiency:.2f} km/kWh")

        # Export data
        st.markdown("#### Export Data")
        if st.button("📥 Download Session Data (CSV)", key="download_btn"):
            df_export = pd.DataFrame({
                'SOC (%)': ev.history['soc'],
                'SOH (%)': ev.history['soh'],
                'Speed (km/h)': ev.history['speed'],
                'Power (kW)': ev.history['power'],
                'Temperature (°C)': ev.history['temperature']
            })
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ev_twin_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_btn"
            )
    else:
        st.info("📊 Start the simulation to collect analytics data")

def main():
    """Main application entry point"""
    initialize_session_state()
    ev = st.session_state.ev_twin

    # Header
    render_header()

    # Sidebar controls
    render_sidebar_controls(ev)

    # Real-time metrics (auto-updates every 0.5s)
    state = render_realtime_metrics()

    st.markdown("---")

    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔋 Battery Management",
        "⚡ Powertrain & Dynamics",
        "🔌 Charging System",
        "📈 Analytics & History"
    ])

    with tab1:
        render_battery_tab(ev, state)

    with tab2:
        render_powertrain_tab(state)

    with tab3:
        render_charging_tab(ev)

    with tab4:
        render_analytics_tab(ev)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 20px;'>
        <p><strong>EV Digital Twin Dashboard</strong> | Real-time OEM/Manufacturer Edition</p>
        <p>Implemented Standards: ISO 15118 • ISO 26262 • CCS Combo • IEC 61851 • AUTOSAR</p>
        <p style='font-size: 0.9em;'>⚡ Auto-refresh: Every 0.5 seconds</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
