# Driving context factors

The same driver state is more dangerous in some conditions than others. CONTEXTRA combines the driver state with these context factors: vehicle speed, road curvature ahead, road roughness, visibility, traffic density, time of day and how long the driver has been driving.

Higher speed leaves less time to react and increases the distance travelled during a lapse of attention. Sharp curves and rough roads demand more steering attention. Fog, heavy rain or darkness reduce visibility and the margin for error. Drowsiness is generally more likely during late-night and early-morning hours and after long periods of continuous driving.

In the built-in simulator and in manual mode these values are simulated. When a CARLA server is connected through the bridge, speed, weather-based visibility and road curvature come from the simulation; road roughness and the night flag are rough proxies.
