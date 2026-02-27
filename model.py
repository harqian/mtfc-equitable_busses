import hashlib
import pathlib

import folium
import pandas as pd

from data_utils import find_data_file


def _route_color(route_id: str) -> str:
    digest = hashlib.md5(route_id.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


if __name__ == "__main__":
    intersections = pd.read_csv(find_data_file("intersections.csv"))
    route_stops = pd.read_csv(find_data_file("simplified_bus_route_stops.csv"), dtype={"route_id": str})

    center_lat = float(pd.to_numeric(intersections["Latitude"], errors="coerce").mean())
    center_lon = float(pd.to_numeric(intersections["Longitude"], errors="coerce").mean())

    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    intersections_layer = folium.FeatureGroup(name="intersections", show=True)
    for row in intersections.itertuples(index=False):
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=1,
            fill=True,
            tooltip=f"{row.Street_Name_1} x {row.Street_Name_2} (zip {row.Zip_Code})",
        ).add_to(intersections_layer)
    intersections_layer.add_to(map_obj)

    route_stops["stop_lat"] = pd.to_numeric(route_stops["stop_lat"], errors="coerce")
    route_stops["stop_lon"] = pd.to_numeric(route_stops["stop_lon"], errors="coerce")
    route_stops["route_stop_order"] = pd.to_numeric(route_stops["route_stop_order"], errors="coerce")
    route_stops["direction_id"] = route_stops["direction_id"].fillna("0").astype(str)
    route_stops = route_stops.dropna(subset=["stop_lat", "stop_lon", "route_stop_order"])

    bus_lines_layer = folium.FeatureGroup(name="bus_lines", show=True)
    for (route_id, direction_id), group in route_stops.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["route_stop_order", "stop_id"])
        coords = list(zip(ordered["stop_lat"], ordered["stop_lon"]))
        if len(coords) < 2:
            continue

        short_name = str(ordered["route_short_name"].iloc[0])
        long_name = str(ordered["route_long_name"].iloc[0])
        folium.PolyLine(
            locations=coords,
            color=_route_color(f"{route_id}-{direction_id}"),
            weight=3,
            opacity=0.8,
            tooltip=f"{short_name} dir {direction_id}: {long_name}",
        ).add_to(bus_lines_layer)
    bus_lines_layer.add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    output = pathlib.Path("sf_intersections_and_bus_lines.html")
    map_obj.save(output)
