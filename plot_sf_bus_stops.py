#!/usr/bin/env -S uv run --with pandas --with geopandas --with folium
import pathlib

import folium
import geopandas as gpd
import pandas as pd
from data_utils import find_data_file

def main() -> None:
    stops_file = find_data_file("stops.txt")
    stops = pd.read_csv(stops_file)

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )

    sf = gpd.read_file(
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/san-francisco.geojson"
    ).to_crs("EPSG:4326")

    sf_outer = gpd.GeoSeries([sf.geometry.union_all()], crs=sf.crs)

    center = sf_outer.iloc[0].centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB positron")

    layer = folium.FeatureGroup(name="Boundary + Bus Stops", show=True)
    folium.GeoJson(
        sf_outer.boundary.__geo_interface__,
        style_function=lambda _: {"color": "black", "weight": 2},
    ).add_to(layer)

    for row in stops_gdf.itertuples(index=False):
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=1,
            weight=1,
            tooltip=f"{row.stop_id}: {row.stop_name}",
        ).add_to(layer)

    layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    output = pathlib.Path("sf_bus_stops_map.html")
    m.save(output)
    print(f"Wrote interactive map: {output.resolve()}")


if __name__ == "__main__":
    main()
