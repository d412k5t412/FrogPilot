import json
import math
import os
import random
import threading

from types import SimpleNamespace

from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.params import Params, UnknownKeyName
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.system.hardware.power_monitoring import VBATT_PAUSE_CHARGING
from panda import ALTERNATIVE_EXPERIENCE

from openpilot.selfdrive.frogpilot.assets.model_manager import DEFAULT_MODEL, DEFAULT_MODEL_NAME, process_model_name
from openpilot.selfdrive.frogpilot.frogpilot_functions import MODELS_PATH

GearShifter = car.CarState.GearShifter
NON_DRIVING_GEARS = [GearShifter.neutral, GearShifter.park, GearShifter.reverse, GearShifter.unknown]

CITY_SPEED_LIMIT = 25                                   # 55mph is typically the minimum speed for highways
CRUISING_SPEED = 5                                      # Roughly the speed cars go when not touching the gas while in drive
MODEL_LENGTH = ModelConstants.IDX_N                     # Minimum length of the model
PLANNER_TIME = ModelConstants.T_IDXS[MODEL_LENGTH - 1]  # Length of time the model projects out for
THRESHOLD = 0.6                                         # 60% chance of condition being true
TO_RADIANS = math.pi / 180                              # Conversion factor from degrees to radians

def get_frogpilot_toggles(block=False):
  toggles_dict = json.loads(Params().get("FrogPilotToggles", block=block))

  if not hasattr(get_frogpilot_toggles, '_frogpilot_toggles_class'):
    attribute_names = list(toggles_dict.keys())
    get_frogpilot_toggles._frogpilot_toggles_class = type(
      'FrogPilotToggles',
      (object,),
      {'__slots__': attribute_names}
    )

  toggle = get_frogpilot_toggles._frogpilot_toggles_class()
  for key in get_frogpilot_toggles._frogpilot_toggles_class.__slots__:
    setattr(toggle, key, toggles_dict.get(key))

  return toggle

def has_prime():
  return Params().get_int("PrimeType") > 0

class FrogPilotVariables:
  def __init__(self):
    self.frogpilot_toggles = SimpleNamespace()

    self.params = Params()
    self.params_memory = Params("/dev/shm/params")

  def update(self, started):
    toggle = self.frogpilot_toggles
    openpilot_installed = self.params.get_bool("HasAcceptedTerms")

    key = "CarParams" if started else "CarParamsPersistent"
    msg_bytes = self.params.get(key, block=openpilot_installed and started)

    if msg_bytes:
      with car.CarParams.from_bytes(msg_bytes) as CP:
        always_on_lateral_set = key == "CarParams" and CP.alternativeExperience & ALTERNATIVE_EXPERIENCE.ALWAYS_ON_LATERAL
        car_make = CP.carName
        car_model = CP.carFingerprint
        has_auto_tune = (car_model == "hyundai" or car_model == "toyota") and CP.lateralTuning.which == "torque"
        has_radar = not CP.radarUnavailable
        is_pid_car = CP.lateralTuning.which == "pid"
        max_acceleration_enabled = key == "CarParams" and CP.alternativeExperience & ALTERNATIVE_EXPERIENCE.RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX
        openpilot_longitudinal = CP.openpilotLongitudinalControl
        pcm_cruise = CP.pcmCruise
    else:
      always_on_lateral_set = False
      car_make = "mock"
      car_model = "mock"
      has_auto_tune = False
      has_radar = False
      is_pid_car = False
      max_acceleration_enabled = False
      openpilot_longitudinal = False
      pcm_cruise = False

    toggle.is_metric = self.params.get_bool("IsMetric")
    distance_conversion = 1 if toggle.is_metric else CV.FOOT_TO_METER
    small_distance_conversion = 1 if toggle.is_metric else CV.INCH_TO_CM
    speed_conversion = CV.KPH_TO_MS if toggle.is_metric else CV.MPH_TO_MS

    toggle.advanced_custom_onroad_ui = self.params.get_bool("AdvancedCustomUI")
    toggle.camera_view = self.params.get_int("CameraView") if toggle.advanced_custom_onroad_ui else 0
    toggle.hide_lead_marker = toggle.advanced_custom_onroad_ui and self.params.get_bool("HideLeadMarker")
    toggle.hide_speed = toggle.advanced_custom_onroad_ui and self.params.get_bool("HideSpeed")
    toggle.hide_speed_ui = toggle.hide_speed and self.params.get_bool("HideSpeedUI")
    toggle.hide_ui_elements = toggle.advanced_custom_onroad_ui and self.params.get_bool("HideUIElements")
    toggle.hide_alerts = toggle.hide_ui_elements and self.params.get_bool("HideAlerts")
    toggle.hide_map_icon = toggle.hide_ui_elements and self.params.get_bool("HideMapIcon")
    toggle.hide_max_speed = toggle.hide_ui_elements and self.params.get_bool("HideMaxSpeed")
    toggle.show_stopping_point = toggle.advanced_custom_onroad_ui and self.params.get_bool("ShowStoppingPoint")
    toggle.show_stopping_point_metrics = toggle.show_stopping_point and self.params.get_bool("ShowStoppingPointMetrics")
    toggle.use_wheel_speed = toggle.advanced_custom_onroad_ui and self.params.get_bool("WheelSpeed")

    toggle.advanced_lateral_tuning = self.params.get_bool("AdvancedLateralTune")
    stock_steer_friction = self.params.get_float("SteerFrictionStock")
    toggle.steer_friction = self.params.get_float("SteerFriction") if toggle.advanced_lateral_tuning else stock_steer_friction
    toggle.use_custom_steer_friction = toggle.steer_friction != stock_steer_friction and not is_pid_car
    stock_steer_kp = self.params.get_float("SteerKPStock")
    toggle.steer_kp = self.params.get_float("SteerKP") if toggle.advanced_lateral_tuning else stock_steer_kp
    toggle.use_custom_kp = toggle.steer_kp != stock_steer_kp and not is_pid_car
    stock_steer_lat_accel_factor = self.params.get_float("SteerLatAccelStock")
    toggle.steer_lat_accel_factor = self.params.get_float("SteerLatAccel") if toggle.advanced_lateral_tuning else stock_steer_lat_accel_factor
    toggle.use_custom_lat_accel_factor = toggle.steer_lat_accel_factor != stock_steer_lat_accel_factor and not is_pid_car
    stock_steer_ratio = self.params.get_float("SteerRatioStock")
    toggle.steer_ratio = self.params.get_float("SteerRatio") if toggle.advanced_lateral_tuning else stock_steer_ratio
    toggle.use_custom_steer_ratio = toggle.steer_ratio != stock_steer_ratio
    toggle.force_auto_tune = toggle.advanced_lateral_tuning and not has_auto_tune and not is_pid_car and self.params.get_bool("ForceAutoTune")
    toggle.force_auto_tune_off = toggle.advanced_lateral_tuning and has_auto_tune and not is_pid_car and self.params.get_bool("ForceAutoTuneOff")
    toggle.taco_tune = toggle.advanced_lateral_tuning and self.params.get_bool("TacoTune")
    toggle.use_turn_desires = toggle.advanced_lateral_tuning and self.params.get_bool("TurnDesires")

    toggle.advanced_longitudinal_tuning = openpilot_longitudinal and self.params.get_bool("AdvancedLongitudinalTune")
    toggle.lead_detection_probability = clip(self.params.get_int("LeadDetectionThreshold") / 100, 0.01, 0.99) if toggle.advanced_longitudinal_tuning else 0.5
    toggle.max_desired_acceleration = clip(self.params.get_float("MaxDesiredAcceleration"), 0.1, 4.0) if toggle.advanced_longitudinal_tuning else 4.0

    toggle.advanced_quality_of_life_driving = self.params.get_bool("AdvancedQOLDriving")
    toggle.force_standstill = toggle.advanced_quality_of_life_driving and self.params.get_bool("ForceStandstill")
    toggle.force_stops = toggle.advanced_quality_of_life_driving and self.params.get_bool("ForceStops")
    toggle.set_speed_offset = self.params.get_int("SetSpeedOffset") * (1 if toggle.is_metric else CV.MPH_TO_KPH) if toggle.advanced_quality_of_life_driving and not pcm_cruise else 0

    toggle.alert_volume_control = self.params.get_bool("AlertVolumeControl")
    toggle.disengage_volume = self.params.get_int("DisengageVolume") if toggle.alert_volume_control else 100
    toggle.engage_volume = self.params.get_int("EngageVolume") if toggle.alert_volume_control else 100
    toggle.prompt_volume = self.params.get_int("PromptVolume") if toggle.alert_volume_control else 100
    toggle.promptDistracted_volume = self.params.get_int("PromptDistractedVolume") if toggle.alert_volume_control else 100
    toggle.refuse_volume = self.params.get_int("RefuseVolume") if toggle.alert_volume_control else 100
    toggle.warningSoft_volume = self.params.get_int("WarningSoftVolume") if toggle.alert_volume_control else 100
    toggle.warningImmediate_volume = max(self.params.get_int("WarningImmediateVolume"), 25) if toggle.alert_volume_control else 100

    toggle.always_on_lateral = always_on_lateral_set and self.params.get_bool("AlwaysOnLateral")
    toggle.always_on_lateral_lkas = toggle.always_on_lateral and car_make != "subaru" and self.params.get_bool("AlwaysOnLateralLKAS")
    toggle.always_on_lateral_main = toggle.always_on_lateral and self.params.get_bool("AlwaysOnLateralMain")
    toggle.always_on_lateral_pause_speed = self.params.get_int("PauseAOLOnBrake") if toggle.always_on_lateral else 0
    toggle.always_on_lateral_status_bar = toggle.always_on_lateral and not self.params.get_bool("HideAOLStatusBar")

    toggle.automatic_updates = self.params.get_bool("AutomaticUpdates")

    toggle.cluster_offset = self.params.get_float("ClusterOffset") if car_make == "toyota" else 1

    toggle.conditional_experimental_mode = openpilot_longitudinal and self.params.get_bool("ConditionalExperimental")
    toggle.conditional_curves = toggle.conditional_experimental_mode and self.params.get_bool("CECurves")
    toggle.conditional_curves_lead = toggle.conditional_curves and self.params.get_bool("CECurvesLead")
    toggle.conditional_lead = toggle.conditional_experimental_mode and self.params.get_bool("CELead")
    toggle.conditional_slower_lead = toggle.conditional_lead and self.params.get_bool("CESlowerLead")
    toggle.conditional_stopped_lead = toggle.conditional_lead and self.params.get_bool("CEStoppedLead")
    toggle.conditional_limit = self.params.get_int("CESpeed") * speed_conversion if toggle.conditional_experimental_mode else 0
    toggle.conditional_limit_lead = self.params.get_int("CESpeedLead") * speed_conversion if toggle.conditional_experimental_mode else 0
    toggle.conditional_navigation = toggle.conditional_experimental_mode and self.params.get_bool("CENavigation")
    toggle.conditional_navigation_intersections = toggle.conditional_navigation and self.params.get_bool("CENavigationIntersections")
    toggle.conditional_navigation_lead = toggle.conditional_navigation and self.params.get_bool("CENavigationLead")
    toggle.conditional_navigation_turns = toggle.conditional_navigation and self.params.get_bool("CENavigationTurns")
    toggle.conditional_model_stop_time = self.params.get_int("CEModelStopTime") if toggle.conditional_experimental_mode else 0
    toggle.conditional_signal = self.params.get_int("CESignalSpeed") if toggle.conditional_experimental_mode else 0
    toggle.conditional_signal_lane_detection = toggle.conditional_signal != 0 and self.params.get_bool("CESignalLaneDetection")
    if toggle.conditional_experimental_mode:
      self.params.put_bool("ExperimentalMode", True)
    toggle.conditional_status_bar = toggle.conditional_experimental_mode and not self.params.get_bool("HideCEMStatusBar")

    toggle.crosstrek_torque = car_make == "subaru" and self.params.get_bool("CrosstrekTorque")

    toggle.current_holiday_theme = self.params.get("CurrentHolidayTheme", encoding='utf-8') if self.params.get_bool("HolidayThemes") else None

    toggle.curve_speed_controller = openpilot_longitudinal and self.params.get_bool("CurveSpeedControl")
    toggle.curve_sensitivity = self.params.get_int("CurveSensitivity") / 100 if toggle.curve_speed_controller else 1
    toggle.turn_aggressiveness = self.params.get_int("TurnAggressiveness") / 100 if toggle.curve_speed_controller else 1
    toggle.disable_curve_speed_smoothing = toggle.curve_speed_controller and self.params.get_bool("DisableCurveSpeedSmoothing")
    toggle.map_turn_speed_controller = toggle.curve_speed_controller and self.params.get_bool("MTSCEnabled")
    toggle.mtsc_curvature_check = toggle.map_turn_speed_controller and self.params.get_bool("MTSCCurvatureCheck")
    toggle.vision_turn_controller = toggle.curve_speed_controller and self.params.get_bool("VisionTurnControl")

    toggle.custom_alerts = self.params.get_bool("CustomAlerts")
    toggle.goat_scream_alert = toggle.current_holiday_theme is None and toggle.custom_alerts and self.params.get_bool("GoatScream")
    toggle.green_light_alert = toggle.custom_alerts and self.params.get_bool("GreenLightAlert")
    toggle.lead_departing_alert = toggle.custom_alerts and self.params.get_bool("LeadDepartingAlert")
    toggle.loud_blindspot_alert = toggle.custom_alerts and self.params.get_bool("LoudBlindspotAlert")
    toggle.speed_limit_changed_alert = toggle.custom_alerts and self.params.get_bool("SpeedLimitChangedAlert")

    toggle.custom_personalities = openpilot_longitudinal and self.params.get_bool("CustomPersonalities")
    toggle.aggressive_profile = toggle.custom_personalities and self.params.get_bool("AggressivePersonalityProfile")
    toggle.aggressive_jerk_acceleration = clip(self.params.get_int("AggressiveJerkAcceleration") / 100, 0.01, 5) if toggle.aggressive_profile else 0.5
    toggle.aggressive_jerk_deceleration = clip(self.params.get_int("AggressiveJerkDeceleration") / 100, 0.01, 5) if toggle.aggressive_profile else 0.5
    toggle.aggressive_jerk_danger = clip(self.params.get_int("AggressiveJerkDanger") / 100, 0.01, 5) if toggle.aggressive_profile else 0.5
    toggle.aggressive_jerk_speed = clip(self.params.get_int("AggressiveJerkSpeed") / 100, 0.01, 5) if toggle.aggressive_profile else 0.5
    toggle.aggressive_jerk_speed_decrease = clip(self.params.get_int("AggressiveJerkSpeedDecrease") / 100, 0.01, 5) if toggle.aggressive_profile else 0.5
    toggle.aggressive_follow = clip(self.params.get_float("AggressiveFollow"), 1, 5) if toggle.aggressive_profile else 1.25
    toggle.standard_profile = toggle.custom_personalities and self.params.get_bool("StandardPersonalityProfile")
    toggle.standard_jerk_acceleration = clip(self.params.get_int("StandardJerkAcceleration") / 100, 0.01, 5) if toggle.standard_profile else 1.0
    toggle.standard_jerk_deceleration = clip(self.params.get_int("StandardJerkDeceleration") / 100, 0.01, 5) if toggle.standard_profile else 1.0
    toggle.standard_jerk_danger = clip(self.params.get_int("StandardJerkDanger") / 100, 0.01, 5) if toggle.standard_profile else 0.5
    toggle.standard_jerk_speed = clip(self.params.get_int("StandardJerkSpeed") / 100, 0.01, 5) if toggle.standard_profile else 1.0
    toggle.standard_jerk_speed_decrease = clip(self.params.get_int("StandardJerkSpeedDecrease") / 100, 0.01, 5) if toggle.standard_profile else 1.0
    toggle.standard_follow = clip(self.params.get_float("StandardFollow"), 1, 5) if toggle.standard_profile else 1.45
    toggle.relaxed_profile = toggle.custom_personalities and self.params.get_bool("RelaxedPersonalityProfile")
    toggle.relaxed_jerk_acceleration = clip(self.params.get_int("RelaxedJerkAcceleration") / 100, 0.01, 5) if toggle.relaxed_profile else 1.0
    toggle.relaxed_jerk_deceleration = clip(self.params.get_int("RelaxedJerkDeceleration") / 100, 0.01, 5) if toggle.relaxed_profile else 1.0
    toggle.relaxed_jerk_danger = clip(self.params.get_int("RelaxedJerkDanger") / 100, 0.01, 5) if toggle.relaxed_profile else 0.5
    toggle.relaxed_jerk_speed = clip(self.params.get_int("RelaxedJerkSpeed") / 100, 0.01, 5) if toggle.relaxed_profile else 1.0
    toggle.relaxed_jerk_speed_decrease = clip(self.params.get_int("RelaxedJerkSpeedDecrease") / 100, 0.01, 5) if toggle.relaxed_profile else 1.0
    toggle.relaxed_follow = clip(self.params.get_float("RelaxedFollow"), 1, 5) if toggle.relaxed_profile else 1.75
    toggle.traffic_profile = toggle.custom_personalities and self.params.get_bool("TrafficPersonalityProfile")
    toggle.traffic_mode_jerk_acceleration = [clip(self.params.get_int("TrafficJerkAcceleration") / 100, 0.01, 5), toggle.aggressive_jerk_acceleration] if toggle.traffic_profile else [0.5, 0.5]
    toggle.traffic_mode_jerk_deceleration = [clip(self.params.get_int("TrafficJerkDeceleration") / 100, 0.01, 5), toggle.aggressive_jerk_deceleration] if toggle.traffic_profile else [0.5, 0.5]
    toggle.traffic_mode_jerk_danger = [clip(self.params.get_int("TrafficJerkDanger") / 100, 0.01, 5), toggle.aggressive_jerk_danger] if toggle.traffic_profile else [1.0, 1.0]
    toggle.traffic_mode_jerk_speed = [clip(self.params.get_int("TrafficJerkSpeed") / 100, 0.01, 5), toggle.aggressive_jerk_speed] if toggle.traffic_profile else [0.5, 0.5]
    toggle.traffic_mode_jerk_speed_decrease = [clip(self.params.get_int("TrafficJerkSpeedDecrease") / 100, 0.01, 5), toggle.aggressive_jerk_speed_decrease] if toggle.traffic_profile else [0.5, 0.5]
    toggle.traffic_mode_t_follow = [clip(self.params.get_float("TrafficFollow"), 0.5, 5), toggle.aggressive_follow] if toggle.traffic_profile else [0.5, 1.0]

    toggle.custom_ui = self.params.get_bool("CustomUI")
    toggle.custom_paths = toggle.custom_ui and self.params.get_bool("CustomPaths")
    toggle.acceleration_path = toggle.custom_paths and self.params.get_bool("AccelerationPath")
    toggle.adjacent_paths = toggle.custom_paths and self.params.get_bool("AdjacentPath")
    toggle.blind_spot_path = toggle.custom_paths and self.params.get_bool("BlindSpotPath")
    toggle.compass = toggle.custom_ui and self.params.get_bool("Compass")
    toggle.dynamic_path_width = toggle.custom_ui and self.params.get_bool("DynamicPathWidth")
    toggle.pedals_on_ui = toggle.custom_ui and self.params.get_bool("PedalsOnUI")
    toggle.dynamic_pedals_on_ui = toggle.pedals_on_ui and self.params.get_bool("DynamicPedalsOnUI")
    toggle.static_pedals_on_ui = toggle.pedals_on_ui and self.params.get_bool("StaticPedalsOnUI")
    toggle.road_name_ui = toggle.custom_ui and self.params.get_bool("RoadNameUI")
    toggle.rotating_wheel = toggle.custom_ui and self.params.get_bool("RotatingWheel")

    toggle.developer_ui = self.params.get_bool("DeveloperUI")
    toggle.border_metrics = toggle.developer_ui and self.params.get_bool("BorderMetrics")
    toggle.blind_spot_metrics = toggle.border_metrics and self.params.get_bool("BlindSpotMetrics")
    toggle.signal_metrics = toggle.border_metrics and self.params.get_bool("SignalMetrics")
    toggle.steering_metrics = toggle.border_metrics and self.params.get_bool("ShowSteering")
    toggle.show_fps = toggle.developer_ui and self.params.get_bool("FPSCounter")
    toggle.lateral_metrics = toggle.developer_ui and self.params.get_bool("LateralMetrics")
    toggle.adjacent_path_metrics = toggle.lateral_metrics and self.params.get_bool("AdjacentPathMetrics")
    toggle.lateral_tuning_metrics = toggle.lateral_metrics and self.params.get_bool("TuningInfo")
    toggle.longitudinal_metrics = toggle.developer_ui and self.params.get_bool("LongitudinalMetrics")
    toggle.adjacent_lead_tracking = has_radar and toggle.longitudinal_metrics and self.params.get_bool("AdjacentLeadsUI")
    toggle.lead_metrics = toggle.longitudinal_metrics and self.params.get_bool("LeadInfo")
    toggle.jerk_metrics = toggle.longitudinal_metrics and self.params.get_bool("JerkInfo")
    toggle.numerical_temp = toggle.developer_ui and self.params.get_bool("NumericalTemp")
    toggle.fahrenheit = toggle.numerical_temp and self.params.get_bool("Fahrenheit")
    toggle.sidebar_metrics = toggle.developer_ui and self.params.get_bool("SidebarMetrics")
    toggle.cpu_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowCPU")
    toggle.gpu_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowGPU")
    toggle.ip_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowIP")
    toggle.memory_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowMemoryUsage")
    toggle.storage_left_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowStorageLeft")
    toggle.storage_used_metrics = toggle.sidebar_metrics and self.params.get_bool("ShowStorageUsed")
    toggle.use_si_metrics = toggle.developer_ui and self.params.get_bool("UseSI")

    toggle.device_management = self.params.get_bool("DeviceManagement")
    device_shutdown_setting = self.params.get_int("DeviceShutdown") if toggle.device_management else 33
    toggle.device_shutdown_time = (device_shutdown_setting - 3) * 3600 if device_shutdown_setting >= 4 else device_shutdown_setting * (60 * 15)
    toggle.increase_thermal_limits = toggle.device_management and self.params.get_bool("IncreaseThermalLimits")
    toggle.low_voltage_shutdown = clip(self.params.get_float("LowVoltageShutdown"), VBATT_PAUSE_CHARGING, 12.5) if toggle.device_management else VBATT_PAUSE_CHARGING
    toggle.no_logging = toggle.device_management and self.params.get_bool("NoLogging")
    toggle.no_uploads = toggle.device_management and self.params.get_bool("NoUploads")
    toggle.offline_mode = toggle.device_management and self.params.get_bool("OfflineMode")

    toggle.experimental_gm_tune = openpilot_longitudinal and car_make == "gm" and self.params.get_bool("ExperimentalGMTune")

    toggle.experimental_mode_via_press = openpilot_longitudinal and self.params.get_bool("ExperimentalModeActivation")
    toggle.experimental_mode_via_distance = toggle.experimental_mode_via_press and self.params.get_bool("ExperimentalModeViaDistance")
    toggle.experimental_mode_via_lkas = not toggle.always_on_lateral_lkas and toggle.experimental_mode_via_press and car_make != "subaru" and self.params.get_bool("ExperimentalModeViaLKAS")
    toggle.experimental_mode_via_tap = toggle.experimental_mode_via_press and self.params.get_bool("ExperimentalModeViaTap")

    toggle.frogsgomoo_tweak = openpilot_longitudinal and car_make == "toyota" and self.params.get_bool("FrogsGoMoosTweak")

    toggle.lane_change_customizations = self.params.get_bool("LaneChangeCustomizations")
    toggle.lane_change_delay = self.params.get_float("LaneChangeTime") if toggle.lane_change_customizations else 0
    toggle.lane_detection_width = self.params.get_float("LaneDetectionWidth") * distance_conversion if toggle.lane_change_customizations else 0
    toggle.lane_detection = toggle.lane_detection_width != 0
    toggle.minimum_lane_change_speed = self.params.get_int("MinimumLaneChangeSpeed") * speed_conversion if toggle.lane_change_customizations else LANE_CHANGE_SPEED_MIN
    toggle.nudgeless = toggle.lane_change_customizations and self.params.get_bool("NudgelessLaneChange")
    toggle.one_lane_change = toggle.lane_change_customizations and self.params.get_bool("OneLaneChange")

    toggle.lateral_tuning = self.params.get_bool("LateralTune")
    toggle.nnff = toggle.lateral_tuning and self.params.get_bool("NNFF")
    toggle.smooth_curve_handling = toggle.lateral_tuning and self.params.get_bool("NNFFLite")

    toggle.long_pitch = openpilot_longitudinal and car_make == "gm" and self.params.get_bool("LongPitch")

    toggle.longitudinal_tuning = openpilot_longitudinal and self.params.get_bool("LongitudinalTune")
    toggle.acceleration_profile = self.params.get_int("AccelerationProfile") if toggle.longitudinal_tuning else 0
    toggle.sport_plus = max_acceleration_enabled and toggle.acceleration_profile == 3
    toggle.deceleration_profile = self.params.get_int("DecelerationProfile") if toggle.longitudinal_tuning else 0
    toggle.human_acceleration = toggle.longitudinal_tuning and self.params.get_bool("HumanAcceleration")
    toggle.human_following = toggle.longitudinal_tuning and self.params.get_bool("HumanFollowing")
    toggle.increased_stopped_distance = self.params.get_int("IncreasedStoppedDistance") * distance_conversion if toggle.longitudinal_tuning else 0

    toggle.model_manager = self.params.get_bool("ModelManagement", block=openpilot_installed)
    available_models = self.params.get("AvailableModels", block=toggle.model_manager, encoding='utf-8') or ""
    available_model_names = self.params.get("AvailableModelsNames", block=toggle.model_manager, encoding='utf-8') or ""
    if toggle.model_manager and available_models:
      if self.params.get_bool("ModelRandomizer"):
        blacklisted_models = (self.params.get("BlacklistedModels", encoding='utf-8') or "").split(',')
        existing_models = [model for model in available_models.split(',') if model not in blacklisted_models and os.path.exists(os.path.join(MODELS_PATH, f"{model}.thneed"))]
        toggle.model = random.choice(existing_models) if existing_models else DEFAULT_MODEL
      else:
        toggle.model = self.params.get("Model", block=True, encoding='utf-8')
    else:
      toggle.model = DEFAULT_MODEL
    if toggle.model in available_models.split(',') and os.path.exists(os.path.join(MODELS_PATH, f"{toggle.model}.thneed")):
      current_model_name = available_model_names.split(',')[available_models.split(',').index(toggle.model)]
      self.params_memory.put("CurrentModelName", current_model_name)
      toggle.part_model_param = process_model_name(current_model_name)
      try:
        self.params.check_key(toggle.part_model_param + "CalibrationParams")
      except UnknownKeyName:
        toggle.part_model_param = ""
    else:
      toggle.model = DEFAULT_MODEL
      toggle.part_model_param = ""
    classic_models = self.params.get("ClassicModels", encoding='utf-8') or ""
    toggle.classic_model = classic_models and toggle.model in classic_models.split(',')
    navigation_models = self.params.get("NavigationModels", encoding='utf-8') or ""
    toggle.navigationless_model = navigation_models and toggle.model not in navigation_models.split(',')
    radarless_models = self.params.get("RadarlessModels", encoding='utf-8') or ""
    toggle.radarless_model = radarless_models and toggle.model in radarless_models.split(',')

    toggle.model_ui = self.params.get_bool("ModelUI")
    toggle.lane_line_width = self.params.get_int("LaneLinesWidth") * small_distance_conversion / 200 if toggle.model_ui else 0.025
    toggle.path_edge_width = self.params.get_int("PathEdgeWidth") if toggle.model_ui else 20
    toggle.path_width = self.params.get_float("PathWidth") * distance_conversion / 2 if toggle.model_ui else 0.9
    toggle.road_edge_width = self.params.get_int("RoadEdgesWidth") * small_distance_conversion / 200 if toggle.model_ui else 0.025
    toggle.unlimited_road_ui_length = toggle.model_ui and self.params.get_bool("UnlimitedLength")

    toggle.new_long_api_gm = openpilot_longitudinal and car_make == "gm" and self.params.get_bool("NewLongAPIGM")
    toggle.new_long_api_hkg = openpilot_longitudinal and car_make == "hyundai" and self.params.get_bool("NewLongAPI")

    toggle.new_toyota_tune = openpilot_longitudinal and car_make == "toyota" and self.params.get_bool("NewToyotaTune")

    toggle.personalize_openpilot = self.params.get_bool("PersonalizeOpenpilot")
    toggle.color_scheme = self.params.get("CustomColors", encoding='utf-8') if toggle.personalize_openpilot else "stock"
    toggle.distance_icons = self.params.get("CustomDistanceIcons", encoding='utf-8') if toggle.personalize_openpilot else "stock"
    toggle.icon_pack = self.params.get("CustomIcons", encoding='utf-8') if toggle.personalize_openpilot else "stock"
    toggle.signal_icons = self.params.get("CustomSignals", encoding='utf-8') if toggle.personalize_openpilot else "stock"
    toggle.sound_pack = self.params.get("CustomSounds", encoding='utf-8') if toggle.personalize_openpilot else "stock"
    toggle.wheel_image = self.params.get("WheelIcon", encoding='utf-8') if toggle.personalize_openpilot else "stock"

    toggle.quality_of_life_lateral = self.params.get_bool("QOLLateral")
    toggle.pause_lateral_below_speed = self.params.get_int("PauseLateralSpeed") * speed_conversion if toggle.quality_of_life_lateral else 0

    toggle.quality_of_life_longitudinal = self.params.get_bool("QOLLongitudinal")
    toggle.custom_cruise_increase = self.params.get_int("CustomCruise") if toggle.quality_of_life_longitudinal and not pcm_cruise else 1
    toggle.custom_cruise_increase_long = self.params.get_int("CustomCruiseLong") if toggle.quality_of_life_longitudinal and not pcm_cruise else 5
    toggle.map_gears = toggle.quality_of_life_longitudinal and self.params.get_bool("MapGears")
    toggle.map_acceleration = toggle.map_gears and self.params.get_bool("MapAcceleration")
    toggle.map_deceleration = toggle.map_gears and self.params.get_bool("MapDeceleration")
    toggle.onroad_distance_button = toggle.quality_of_life_longitudinal and self.params.get_bool("OnroadDistanceButton")
    toggle.pause_lateral_below_signal = toggle.pause_lateral_below_speed != 0 and self.params.get_bool("PauseLateralOnSignal")
    toggle.reverse_cruise_increase = toggle.quality_of_life_longitudinal and pcm_cruise and self.params.get_bool("ReverseCruise")

    toggle.quality_of_life_visuals = self.params.get_bool("QOLVisuals")
    toggle.big_map = toggle.quality_of_life_visuals and self.params.get_bool("BigMap")
    toggle.full_map = toggle.big_map and self.params.get_bool("FullMap")
    toggle.driver_camera_in_reverse = toggle.quality_of_life_visuals and self.params.get_bool("DriverCamera")
    toggle.map_style = self.params.get_int("MapStyle") if toggle.quality_of_life_visuals else 0
    toggle.standby_mode = toggle.quality_of_life_visuals and self.params.get_bool("StandbyMode")
    toggle.stopped_timer = toggle.quality_of_life_visuals and self.params.get_bool("StoppedTimer")

    toggle.random_events = self.params.get_bool("RandomEvents")

    toggle.screen_management = self.params.get_bool("ScreenManagement")
    toggle.screen_brightness = self.params.get_int("ScreenBrightness") if toggle.screen_management else 101
    toggle.screen_brightness_onroad = self.params.get_int("ScreenBrightnessOnroad") if toggle.screen_management else 101
    toggle.screen_recorder = toggle.screen_management and self.params.get_bool("ScreenRecorder")
    toggle.screen_timeout = self.params.get_int("ScreenTimeout") if toggle.screen_management else 30
    toggle.screen_timeout_onroad = self.params.get_int("ScreenTimeoutOnroad") if toggle.screen_management else 10

    toggle.sng_hack = openpilot_longitudinal and car_make == "toyota" and self.params.get_bool("SNGHack")

    toggle.speed_limit_controller = openpilot_longitudinal and self.params.get_bool("SpeedLimitController")
    toggle.force_mph_dashboard = toggle.speed_limit_controller and self.params.get_bool("ForceMPHDashboard")
    toggle.map_speed_lookahead_higher = self.params.get_int("SLCLookaheadHigher") if toggle.speed_limit_controller else 0
    toggle.map_speed_lookahead_lower = self.params.get_int("SLCLookaheadLower") if toggle.speed_limit_controller else 0
    toggle.set_speed_limit = toggle.speed_limit_controller and self.params.get_bool("SetSpeedLimit")
    toggle.show_speed_limit_offset = toggle.speed_limit_controller and self.params.get_bool("ShowSLCOffset")
    toggle.show_speed_limit_offset_ui = toggle.show_speed_limit_offset and self.params.get_bool("ShowSLCOffsetUI")
    slc_fallback_method = self.params.get_int("SLCFallback")
    toggle.slc_fallback_experimental_mode = toggle.speed_limit_controller and slc_fallback_method == 1
    toggle.slc_fallback_previous_speed_limit = toggle.speed_limit_controller and slc_fallback_method == 2
    toggle.slc_fallback_set_speed = toggle.speed_limit_controller and slc_fallback_method == 0
    toggle.speed_limit_confirmation_higher = toggle.speed_limit_controller and self.params.get_bool("SLCConfirmationHigher")
    toggle.speed_limit_confirmation_lower = toggle.speed_limit_controller and self.params.get_bool("SLCConfirmationLower")
    toggle.speed_limit_controller_override = self.params.get_int("SLCOverride") if toggle.speed_limit_controller else 0
    toggle.speed_limit_controller_override_manual = toggle.speed_limit_controller_override == 1
    toggle.speed_limit_controller_override_set_speed = toggle.speed_limit_controller_override == 2
    toggle.speed_limit_offset1 = self.params.get_int("Offset1") * speed_conversion if toggle.speed_limit_controller else 0
    toggle.speed_limit_offset2 = self.params.get_int("Offset2") * speed_conversion if toggle.speed_limit_controller else 0
    toggle.speed_limit_offset3 = self.params.get_int("Offset3") * speed_conversion if toggle.speed_limit_controller else 0
    toggle.speed_limit_offset4 = self.params.get_int("Offset4") * speed_conversion if toggle.speed_limit_controller else 0
    toggle.speed_limit_priority1 = self.params.get("SLCPriority1", encoding='utf-8') if toggle.speed_limit_controller else None
    toggle.speed_limit_priority2 = self.params.get("SLCPriority2", encoding='utf-8') if toggle.speed_limit_controller else None
    toggle.speed_limit_priority3 = self.params.get("SLCPriority3", encoding='utf-8') if toggle.speed_limit_controller else None
    toggle.speed_limit_priority_highest = toggle.speed_limit_priority1 == "Highest"
    toggle.speed_limit_priority_lowest = toggle.speed_limit_priority1 == "Lowest"
    toggle.speed_limit_vienna = toggle.speed_limit_controller and self.params.get_bool("UseVienna")

    toggle.tethering_config = self.params.get_int("TetheringEnabled")

    toggle.toyota_doors = car_make == "toyota" and self.params.get_bool("ToyotaDoors")
    toggle.lock_doors = toggle.toyota_doors and self.params.get_bool("LockDoors")
    toggle.unlock_doors = toggle.toyota_doors and self.params.get_bool("UnlockDoors")

    toggle.volt_sng = car_model == "CHEVROLET_VOLT" and self.params.get_bool("VoltSNG")

    self.params.put("FrogPilotToggles", json.dumps(toggle.__dict__))
    self.params_memory.remove("FrogPilotTogglesUpdated")
