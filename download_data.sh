#!/bin/sh
# Download the Natural Earth 1:110m land shapefile into data/ne_110m_land/.
# Safe to run again: does nothing when the shapefile is already there.
set -eu

cd "$(dirname "$0")"

url="https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
dir="data/ne_110m_land"
zip="data/ne_110m_land.zip"

if [ -f "$dir/ne_110m_land.shp" ]; then
    echo "$dir/ne_110m_land.shp exists, nothing to do"
    exit 0
fi

mkdir -p "$dir"
curl -fL -o "$zip" "$url"
unzip -o "$zip" -d "$dir"
rm "$zip"
echo "wrote $dir/ne_110m_land.shp"
