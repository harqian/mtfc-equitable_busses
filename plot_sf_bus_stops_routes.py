#!/usr/bin/env -S uv run --with pandas --with geopandas --with folium
import pathlib

import folium
import geopandas as gpd
import pandas as pd
from branca.element import MacroElement, Template
from shapely.geometry import LineString

from data_utils import find_data_file



def add_route_toggle_control(map_obj: folium.Map) -> None:
    template = Template(
        """
        {% macro html(this, kwargs) %}
        <div id="route-bulk-toggle"
             style="position: fixed; top: 10px; left: 55px; z-index: 9999;
                    background: white; border: 1px solid #999; border-radius: 4px;
                    padding: 6px; font-family: sans-serif; font-size: 12px;">
          <button type="button" onclick="setAllRoutes(true)">Show all routes</button>
          <button type="button" onclick="setAllRoutes(false)">Hide all routes</button>
        </div>
        <script>
          function setAllRoutes(show) {
            const labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
            labels.forEach((label) => {
              const text = label.textContent || '';
              if (!text.includes('Route ')) return;
              const input = label.querySelector('input[type="checkbox"]');
              if (!input) return;
              if (input.checked !== show) input.click();
            });
          }
        </script>
        {% endmacro %}
        """
    )
    macro = MacroElement()
    macro._template = template
    map_obj.get_root().add_child(macro)


def main() -> None:
    stops = pd.read_csv(find_data_file('stops.txt'))
    trips = pd.read_csv(find_data_file('trips.txt'))
    routes = pd.read_csv(find_data_file('routes.txt'))
    shapes = pd.read_csv(find_data_file('shapes.txt'))

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']),
        crs='EPSG:4326',
    )

    shapes = shapes.sort_values(['shape_id', 'shape_pt_sequence'])
    shape_geoms = []
    for shape_id, group in shapes.groupby('shape_id', sort=False):
        coords = list(zip(group['shape_pt_lon'], group['shape_pt_lat']))
        if len(coords) >= 2:
            shape_geoms.append({'shape_id': shape_id, 'geometry': LineString(coords)})
    shapes_gdf = gpd.GeoDataFrame(shape_geoms, crs='EPSG:4326')

    trip_stats = trips.groupby('route_id').agg(
        trip_count=('trip_id', 'nunique'),
        shape_count=('shape_id', 'nunique'),
        direction_count=('direction_id', 'nunique'),
    )
    headsigns = (
        trips.dropna(subset=['trip_headsign'])
        .groupby('route_id')['trip_headsign']
        .agg(lambda s: ' | '.join(sorted(s.astype(str).unique())[:3]))
        .rename('sample_headsigns')
    )
    route_meta = routes.merge(trip_stats, on='route_id', how='left').merge(
        headsigns, on='route_id', how='left'
    )
    route_meta['trip_count'] = route_meta['trip_count'].fillna(0).astype(int)
    route_meta['shape_count'] = route_meta['shape_count'].fillna(0).astype(int)
    route_meta['direction_count'] = route_meta['direction_count'].fillna(0).astype(int)
    route_meta['sample_headsigns'] = route_meta['sample_headsigns'].fillna('')
    route_meta['route_short_name'] = route_meta['route_short_name'].fillna('').astype(str)
    route_meta['route_long_name'] = route_meta['route_long_name'].fillna('').astype(str)
    route_meta['route_desc'] = route_meta['route_desc'].fillna('').astype(str)

    route_shapes = (
        trips[['route_id', 'shape_id']]
        .dropna(subset=['shape_id'])
        .drop_duplicates()
        .merge(shapes_gdf, on='shape_id', how='inner')
        .merge(
            route_meta[
                [
                    'route_id',
                    'route_short_name',
                    'route_long_name',
                    'route_desc',
                    'route_color',
                    'trip_count',
                    'shape_count',
                    'direction_count',
                    'sample_headsigns',
                ]
            ],
            on='route_id',
            how='left',
        )
    )
    route_shapes_gdf = gpd.GeoDataFrame(route_shapes, geometry='geometry', crs='EPSG:4326')

    sf = gpd.read_file(
        'https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/san-francisco.geojson'
    ).to_crs('EPSG:4326')
    sf_outer = gpd.GeoSeries([sf.geometry.union_all()], crs=sf.crs)

    center = sf_outer.iloc[0].centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles='CartoDB positron')

    stops_layer = folium.FeatureGroup(name='Boundary + Bus Stops', show=True)
    folium.GeoJson(
        sf_outer.boundary.__geo_interface__,
        style_function=lambda _: {'color': 'black', 'weight': 2},
    ).add_to(stops_layer)
    for row in stops_gdf.itertuples(index=False):
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=2,
            color='red',
            fill=True,
            fill_opacity=1,
            weight=1,
            tooltip=f'{row.stop_id}: {row.stop_name}',
        ).add_to(stops_layer)
    stops_layer.add_to(m)

    for route_id, group in route_shapes_gdf.groupby('route_id', sort=True):
        first = group.iloc[0]
        short_name = first['route_short_name'].strip()
        long_name = first['route_long_name'].strip()
        layer_name = f'Route {short_name}' if short_name else f'Route {route_id}'
        if long_name:
            layer_name = f'{layer_name} - {long_name}'

        raw_color = str(first['route_color']) if pd.notna(first['route_color']) else ''
        color = f'#{raw_color}' if len(raw_color) == 6 else '#1f77b4'
        route_layer = folium.FeatureGroup(name=layer_name, show=False)
        folium.GeoJson(
            group.__geo_interface__,
            style_function=lambda _, c=color: {'color': c, 'weight': 2, 'opacity': 0.7},
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    'route_id',
                    'route_short_name',
                    'route_long_name',
                    'trip_count',
                    'shape_count',
                    'direction_count',
                    'sample_headsigns',
                ],
                aliases=[
                    'Route ID',
                    'Short Name',
                    'Long Name',
                    'Trips',
                    'Shapes',
                    'Directions',
                    'Headsigns (sample)',
                ],
            ),
        ).add_to(route_layer)
        route_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    add_route_toggle_control(m)

    output = pathlib.Path('sf_bus_stops_routes_map.html')
    m.save(output)
    print(f'Wrote interactive map: {output.resolve()}')


if __name__ == '__main__':
    main()
