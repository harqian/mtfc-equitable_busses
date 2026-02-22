import pandas as pd
from data_utils import find_data_file
import folium
import geopandas as gpd
import pathlib

if __name__ == "__main__":
    intersections = pd.read_csv(find_data_file("intersections.csv"))
    print(intersections)
    intersections_gdf = gpd.GeoDataFrame(
        intersections,
        geometry=gpd.points_from_xy(intersections["Longitude"], intersections["Latitude"])
    )
    sf = gpd.read_file(
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/san-francisco.geojson"
    ).to_crs("EPSG:4326")
    sf_outer = gpd.GeoSeries([sf.geometry.union_all()], crs=sf.crs)
    center = sf_outer.iloc[0].centroid
    
    map = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB positron")
    layer = folium.FeatureGroup(name="intersections", show=True)
    for row in intersections_gdf.itertuples(index=False):
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=1,
            fill=True,
            tooltip=f"{row.Street_Name_1} x {row.Street_Name_2} (zip {row.Zip_Code})"
        ).add_to(layer)

    layer.add_to(map)
    folium.LayerControl(collapsed=False).add_to(map)

    output = pathlib.Path("sf_intersections.html")
    map.save(output)


