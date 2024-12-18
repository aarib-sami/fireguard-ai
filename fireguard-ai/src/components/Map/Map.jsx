import React, { useState } from 'react';
import ReactMapGl from "react-map-gl";

function Map() {
  const [viewport, setViewport] = useState({
    latitude: 54.243459,
    longitude: -9.073448,
    zoom: 1
  });

  return (
    <div className="map-container">
      <ReactMapGl
        initialViewState={viewport}
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_API_KEY}
        mapStyle="mapbox://styles/mapbox/satellite-v9"
        onMove={evt => setViewport(evt.viewState)}
      >
      </ReactMapGl>
    </div>
  );
}

export default Map;