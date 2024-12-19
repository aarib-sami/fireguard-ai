import React, { useState } from 'react';
import ReactMapGl from 'react-map-gl';
import "mapbox-gl/dist/mapbox-gl.css";
import './Map.css';
import Button from '@mui/material/Button';


function Map(props) {
  const [viewport, setViewport] = useState({
    latitude: 54.243459,
    longitude: -9.073448,
    zoom: 1,
  });

  const [moving, setMoving] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  function handleMove(evt)
  {
    setViewport(evt.viewState)
  }

  function handleMapClick(evt) {
    setZoomed(true);
    const { lngLat: { lat: latitude, lng: longitude } } = evt;
    setViewport({
      latitude,
      longitude,
      zoom: 14,
    });
    props.enablePrediction();
  };

  function resetMap(){
    setViewport({
      latitude: 54.243459,
      longitude: -9.073448,
      zoom: 1,
    });
    props.disablePrediction();
  }

  return (
    <div className="map-container">
      <ReactMapGl
        {...viewport}
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_API_KEY}
        mapStyle="mapbox://styles/mapbox/satellite-v9"
        onMove={!moving && handleMove}
        onClick={handleMapClick}
      >
      </ReactMapGl>
      <Button variant="contained" size="small" className="reset-button" onClick={resetMap}>Reset Map</Button>
    </div>
  );
}

export default Map;
